"""
Fine-tune Whisper for Morse Code Recognition - Optimized for GPU Utilization

This script fine-tunes OpenAI's Whisper model with optimizations for consistent GPU usage.
"""

import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from tqdm import tqdm
import multiprocessing as mp
import platform  # Add platform import

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)
from datasets import Dataset, DatasetDict, Audio
import evaluate
import librosa
import soundfile as sf

# Import the audio chunking module if needed
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from audio_chunking import WhisperAudioChunker


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper fine-tuning."""
    
    processor: WhisperProcessor
    decoder_start_token_id: int
    
    def __call__(self, features: list[Dict[str, Any]]) -> Dict[str, Any]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad inputs
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Remove decoder_start_token_id if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        
        return batch


def prepare_dataset_optimized(data_dir: Path, processor: WhisperProcessor, 
                            use_chunking: bool = False,
                            chunk_strategy: str = "sequential",
                            max_samples: int = None,
                            num_workers: int = None) -> DatasetDict:
    """Prepare dataset for training - optimized version with better data loading."""
    datasets = {}
    
    # Auto-detect number of workers - Windows needs special handling
    if num_workers is None:
        if platform.system() == 'Windows':
            # Windows has issues with multiprocessing in DataLoader
            num_workers = 0
            print("Windows detected: Using single-process data loading for stability")
        else:
            num_workers = min(mp.cpu_count() - 1, 8)
    
    print(f"Using {num_workers} data loader workers")
    
    # Check if dataset has duration information
    metadata_path = data_dir / 'metadata.csv'
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if 'duration' in metadata.columns:
            avg_duration = metadata['duration'].mean()
            max_duration = metadata['duration'].max()
            print(f"Dataset statistics: avg duration {avg_duration:.1f}s, max {max_duration:.1f}s")
            
            # Recommend chunking if needed
            if max_duration > 30 and not use_chunking:
                print("Warning: Dataset contains audio > 30s but chunking is disabled!")
    
    audio_dir = data_dir / 'audio'
    
    def load_audio_fast(audio_path):
        """Faster audio loading using soundfile."""
        audio, sr = sf.read(audio_path)
        # Resample if needed
        if sr != 16000:
            import resampy
            audio = resampy.resample(audio, sr, 16000)
        return audio
    
    def preprocess_function(examples):
        """Process audio files - optimized version."""
        audio_files = examples["audio_file"]
        texts = examples["text"]
        
        # Process each sample
        input_features = []
        labels = []
        
        for audio_file, text in zip(audio_files, texts):
            # Load audio - using soundfile for speed
            audio_path = str(audio_dir / audio_file)
            try:
                audio = load_audio_fast(audio_path)
            except:
                # Fallback to librosa if soundfile fails
                audio, sr = librosa.load(audio_path, sr=16000)
            
            # Pad to 30 seconds if needed
            target_length = 30 * 16000  # 480,000 samples
            if len(audio) < target_length:
                pad_length = target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            elif len(audio) > target_length:
                audio = audio[:target_length]
            
            # Process audio
            features = processor.feature_extractor(
                audio, 
                sampling_rate=16000,
                return_tensors="np"
            ).input_features[0]
            
            input_features.append(features)
            
            # Process text
            # Handle NaN/empty text
            if pd.isna(text) or text is None or text == "":
                text = ""
                
            if text.strip():
                full_text = f"<|morse|> {text} <|endoftext|>"
            else:
                # For noise-only samples
                full_text = "<|noise|> <|endoftext|>"
            
            label_ids = processor.tokenizer(full_text).input_ids
            labels.append(label_ids)
        
        return {
            "input_features": input_features,
            "labels": labels
        }
    
    for split in ['train', 'validation', 'test']:
        print(f"Loading {split} dataset...")
        
        # Check if split exists
        split_file = data_dir / 'splits' / f'{split}.csv'
        if not split_file.exists():
            print(f"Warning: {split_file} not found, skipping...")
            continue
        
        # Load metadata
        split_df = pd.read_csv(split_file)
        # Ensure text column is string type
        split_df['text'] = split_df['text'].fillna('')
        
        # Limit samples if requested
        if max_samples and split == 'train':
            split_df = split_df.head(max_samples)
        
        print(f"Creating dataset from {len(split_df)} samples...")
        
        # Create dataset from pandas - just stores metadata
        dataset = Dataset.from_pandas(split_df[['audio_file', 'text']])
        
        # Process with larger batches and multiple workers
        batch_size = 64 if split == 'train' else 16
        # Only use multiprocessing for dataset preparation on non-Windows or if explicitly requested
        dataset_num_proc = num_workers if (platform.system() != 'Windows' and split == 'train') else 1
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=batch_size,
            num_proc=dataset_num_proc,  # Use adjusted number of processes
            remove_columns=dataset.column_names,
            desc=f"Processing {split} audio",
            load_from_cache_file=True  # Enable caching
        )
        
        # Set format for PyTorch
        dataset.set_format(type='torch', columns=['input_features', 'labels'])
        
        datasets[split] = dataset
    
    return DatasetDict(datasets)


def compute_metrics(eval_preds, tokenizer, metric):
    """Compute WER metric."""
    pred_ids = eval_preds.predictions
    label_ids = eval_preds.label_ids
    
    # Replace -100 with pad token id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Filter out noise-only samples (empty labels) for WER calculation
    valid_pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
    
    if valid_pairs:
        valid_preds, valid_labels = zip(*valid_pairs)
        wer = 100 * metric.compute(predictions=valid_preds, references=valid_labels)
    else:
        wer = 0.0
    
    # Also compute accuracy on noise samples
    noise_correct = sum(1 for p, l in zip(pred_str, label_str) 
                       if not l.strip() and not p.strip())
    noise_total = sum(1 for l in label_str if not l.strip())
    noise_accuracy = noise_correct / noise_total if noise_total > 0 else 1.0
    
    return {
        "wer": wer,
        "noise_accuracy": noise_accuracy * 100,
        "num_morse_samples": len(valid_pairs),
        "num_noise_samples": noise_total
    }


def fine_tune_whisper_optimized(args):
    """Main fine-tuning function - optimized for GPU utilization."""
    print(f"Fine-tuning Whisper model: {args.model_name}")
    print("Using optimized training configuration for better GPU utilization")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load processor and model
    print("Loading model and processor...")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Add special tokens for morse code and noise
    special_tokens = ["<|morse|>", "<|noise|>"]
    processor.tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))
    
    # Configure for morse code
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    
    # Move model to GPU
    model.to(device)
    
    # Prepare dataset
    print("Preparing dataset...")
    print(f"Chunking: {'enabled' if args.use_chunking else 'disabled'}")
    
    # Use optimized dataset preparation
    dataset = prepare_dataset_optimized(
        Path(args.data_dir), 
        processor, 
        use_chunking=args.use_chunking,
        chunk_strategy=args.chunk_strategy,
        max_samples=args.max_samples,
        num_workers=args.dataloader_num_workers
    )
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Load WER metric
    metric = evaluate.load("wer")
    
    # Optimized training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 1,
        gradient_checkpointing=True,
        fp16=args.fp16,
        bf16=args.bf16,  # BF16 can be more stable than FP16
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        save_total_limit=3,
        # Optimization settings - compatible with transformers 4.36.2
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
        # Note: dataloader_persistent_workers and dataloader_prefetch_factor 
        # are not available in transformers 4.36.2
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("validation"),
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, processor.tokenizer, metric
        ),
        tokenizer=processor.feature_extractor,
    )
    
    # Train
    print("Starting training with optimized settings...")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Data loader workers: {args.dataloader_num_workers}")
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print("Saving model...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    # Save training configuration
    train_config = {
        'model_name': args.model_name,
        'use_chunking': args.use_chunking,
        'chunk_strategy': args.chunk_strategy if args.use_chunking else None,
        'special_tokens': special_tokens,
        'training_samples': len(dataset.get("train", [])),
        'validation_samples': len(dataset.get("validation", [])),
        'optimization_settings': {
            'dataloader_num_workers': args.dataloader_num_workers,
            'batch_size': args.batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'fp16': args.fp16,
            'bf16': args.bf16
        }
    }
    with open(Path(args.output_dir) / "training_config.json", "w") as f:
        json.dump(train_config, f, indent=2)
    
    # Evaluate on test set if available
    if "test" in dataset:
        print("Evaluating on test set...")
        test_results = trainer.evaluate(
            eval_dataset=dataset["test"],
            metric_key_prefix="test"
        )
        
        # Save test results
        with open(Path(args.output_dir) / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Test WER: {test_results.get('test_wer', 'N/A'):.2f}%")
        print(f"Test Noise Accuracy: {test_results.get('test_noise_accuracy', 'N/A'):.2f}%")
    
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for morse code - Optimized")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-small",
        help="Pretrained Whisper model name"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the dataset"
    )
    
    # Chunking arguments
    parser.add_argument(
        "--use_chunking",
        action="store_true",
        help="Enable chunking for audio files > 30 seconds (not needed for properly prepared datasets)"
    )
    parser.add_argument(
        "--chunk_strategy",
        type=str,
        default="sequential",
        choices=["sequential", "sliding", "random"],
        help="Strategy for chunking long audio files (only used if --use_chunking)"
    )
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 training (recommended for RTX 4080)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    
    # Optimization arguments
    default_workers = 0 if platform.system() == 'Windows' else 4
    parser.add_argument("--dataloader_num_workers", type=int, default=default_workers,
                       help=f"Number of data loader workers (default: {default_workers})")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fine-tune
    fine_tune_whisper_optimized(args)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    if platform.system() == 'Windows':
        mp.freeze_support()
    main() 