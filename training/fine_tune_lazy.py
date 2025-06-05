"""
Fine-tune Whisper for Morse Code Recognition - True Lazy Loading Version

This script uses a custom dataset class that loads audio on-demand,
keeping memory usage minimal even with large datasets.
"""

import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import multiprocessing as mp
import platform

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
from torch.utils.data import Dataset as TorchDataset

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


class LazyWhisperDataset(TorchDataset):
    """
    A truly lazy dataset that loads audio on-demand.
    Only stores file paths and metadata in memory.
    """
    
    def __init__(self, 
                 data_dir: Path,
                 split: str,
                 processor: WhisperProcessor,
                 use_chunking: bool = False,
                 chunk_strategy: str = "sequential",
                 max_samples: Optional[int] = None):
        """
        Initialize the lazy dataset.
        
        Args:
            data_dir: Root data directory
            split: 'train', 'validation', or 'test'
            processor: Whisper processor
            use_chunking: Whether to use chunking for long audio
            chunk_strategy: Strategy for chunking if enabled
            max_samples: Maximum number of samples to use
        """
        self.data_dir = data_dir
        self.audio_dir = data_dir / 'audio'
        self.processor = processor
        self.use_chunking = use_chunking
        self.chunk_strategy = chunk_strategy
        self.sampling_rate = 16000
        
        # Load only metadata - no audio loading here!
        split_file = data_dir / 'splits' / f'{split}.csv'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        self.metadata = pd.read_csv(split_file)
        self.metadata['text'] = self.metadata['text'].fillna('')
        
        # Limit samples if requested
        if max_samples:
            self.metadata = self.metadata.head(max_samples)
            
        # Initialize chunker if needed
        if use_chunking:
            self.chunker = WhisperAudioChunker(overlap_seconds=2.0)
            self._prepare_chunks()
        else:
            self.samples = self.metadata
            
        print(f"Initialized lazy dataset with {len(self.samples)} samples for {split}")
        
        # Check for long audio files
        if 'duration' in self.metadata.columns and not use_chunking:
            long_samples = self.metadata[self.metadata['duration'] > 30]
            if len(long_samples) > 0:
                print(f"Warning: {len(long_samples)} samples exceed 30 seconds!")
    
    def _prepare_chunks(self):
        """Prepare chunk metadata without loading audio."""
        chunked_samples = []
        
        for idx, row in self.metadata.iterrows():
            if row.get('duration', 30) <= 30:
                chunked_samples.append({
                    'original_idx': idx,
                    'audio_file': row['audio_file'],
                    'text': row['text'],
                    'chunk_idx': 0,
                    'chunk_start': 0.0,
                    'chunk_end': row.get('duration', 30),
                    'is_chunked': False
                })
            else:
                # Create chunk metadata for long audio
                num_chunks = int(np.ceil(row['duration'] / 28))
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * 28
                    chunk_end = min(chunk_start + 30, row['duration'])
                    
                    chunked_samples.append({
                        'original_idx': idx,
                        'audio_file': row['audio_file'],
                        'text': row['text'],
                        'chunk_idx': chunk_idx,
                        'chunk_start': chunk_start,
                        'chunk_end': chunk_end,
                        'is_chunked': True
                    })
        
        self.samples = pd.DataFrame(chunked_samples)
    
    def __len__(self):
        return len(self.samples)
    
    def _load_audio_fast(self, audio_path):
        """Fast audio loading using soundfile."""
        try:
            audio, sr = sf.read(audio_path)
            if sr != self.sampling_rate:
                import resampy
                audio = resampy.resample(audio, sr, self.sampling_rate)
            return audio
        except:
            # Fallback to librosa
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            return audio
    
    def __getitem__(self, idx):
        """
        Load and process a single sample on-demand.
        This is called by the DataLoader when it needs a sample.
        """
        row = self.samples.iloc[idx]
        
        # Load audio only when needed
        audio_path = self.audio_dir / row['audio_file']
        audio = self._load_audio_fast(str(audio_path))
        
        # Handle chunking if needed
        if self.use_chunking and row.get('is_chunked', False):
            # Extract chunk
            start_sample = int(row['chunk_start'] * self.sampling_rate)
            end_sample = int(row['chunk_end'] * self.sampling_rate)
            audio = audio[start_sample:end_sample]
            
            # Align transcript
            text_value = row['text'] if not pd.isna(row['text']) else ""
            if hasattr(self, 'chunker'):
                total_duration = len(audio) / self.sampling_rate
                text = self.chunker._align_transcript(
                    str(text_value), row['chunk_start'], row['chunk_end'], total_duration
                )
            else:
                text = text_value
        else:
            text = row['text'] if not pd.isna(row['text']) else ""
        
        # Pad audio to 30 seconds
        target_length = 30 * self.sampling_rate
        if len(audio) < target_length:
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
        elif len(audio) > target_length:
            audio = audio[:target_length]
        
        # Process audio features
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_features[0]
        
        # Process text
        if pd.isna(text) or text is None:
            text = ""
        else:
            text = str(text)
            
        if text.strip():
            full_text = f"<|morse|> {text} <|endoftext|>"
        else:
            full_text = "<|noise|> <|endoftext|>"
        
        labels = self.processor.tokenizer(full_text).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels
        }


def create_lazy_datasets(data_dir: Path, 
                        processor: WhisperProcessor,
                        use_chunking: bool = False,
                        chunk_strategy: str = "sequential",
                        max_samples: Optional[int] = None) -> Dict[str, LazyWhisperDataset]:
    """Create lazy datasets for all splits."""
    datasets = {}
    
    # Check dataset statistics
    metadata_path = data_dir / 'metadata.csv'
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if 'duration' in metadata.columns:
            avg_duration = metadata['duration'].mean()
            max_duration = metadata['duration'].max()
            print(f"Dataset statistics: avg duration {avg_duration:.1f}s, max {max_duration:.1f}s")
            
            if max_duration > 30 and not use_chunking:
                print("Warning: Dataset contains audio > 30s but chunking is disabled!")
    
    for split in ['train', 'validation', 'test']:
        split_file = data_dir / 'splits' / f'{split}.csv'
        if split_file.exists():
            datasets[split] = LazyWhisperDataset(
                data_dir=data_dir,
                split=split,
                processor=processor,
                use_chunking=use_chunking,
                chunk_strategy=chunk_strategy,
                max_samples=max_samples if split == 'train' else None
            )
        else:
            print(f"Warning: {split_file} not found, skipping...")
    
    return datasets


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


def fine_tune_whisper_lazy(args):
    """Main fine-tuning function with lazy loading."""
    print(f"Fine-tuning Whisper model: {args.model_name}")
    print("Using LAZY LOADING for minimal memory usage")
    
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
    
    # Create lazy datasets
    print("Creating lazy datasets...")
    print(f"Chunking: {'enabled' if args.use_chunking else 'disabled'}")
    
    datasets = create_lazy_datasets(
        Path(args.data_dir),
        processor,
        use_chunking=args.use_chunking,
        chunk_strategy=args.chunk_strategy,
        max_samples=args.max_samples
    )
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Load WER metric
    metric = evaluate.load("wer")
    
    # Training arguments - optimized for lazy loading
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
        bf16=args.bf16,
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
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    # Create trainer with lazy datasets
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("validation"),
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, processor.tokenizer, metric
        ),
        tokenizer=processor.feature_extractor,
    )
    
    # Train
    print("Starting training with lazy loading...")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Data loader workers: {args.dataloader_num_workers}")
    print("Memory usage: Only loading audio as needed!")
    
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
        'training_samples': len(datasets.get("train", [])),
        'validation_samples': len(datasets.get("validation", [])),
        'lazy_loading': True,
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
    if "test" in datasets:
        print("Evaluating on test set...")
        test_results = trainer.evaluate(
            eval_dataset=datasets["test"],
            metric_key_prefix="test"
        )
        
        # Save test results
        with open(Path(args.output_dir) / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Test WER: {test_results.get('test_wer', 'N/A'):.2f}%")
        print(f"Test Noise Accuracy: {test_results.get('test_noise_accuracy', 'N/A'):.2f}%")
    
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper for morse code - True Lazy Loading Version"
    )
    
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
        help="Enable chunking for audio files > 30 seconds"
    )
    parser.add_argument(
        "--chunk_strategy",
        type=str,
        default="sequential",
        choices=["sequential", "sliding", "random"],
        help="Strategy for chunking long audio files"
    )
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    
    # Optimization arguments
    default_workers = 0 if platform.system() == 'Windows' else 4
    parser.add_argument(
        "--dataloader_num_workers", 
        type=int, 
        default=default_workers,
        help=f"Number of data loader workers (default: {default_workers})"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fine-tune
    fine_tune_whisper_lazy(args)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    if platform.system() == 'Windows':
        mp.freeze_support()
    main() 