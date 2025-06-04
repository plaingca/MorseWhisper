"""
Fine-tune Whisper for Morse Code Recognition

This script fine-tunes OpenAI's Whisper model on morse code audio data.
It uses the Hugging Face transformers library for training.
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


class MorseWhisperDataset:
    """Dataset class for morse code audio."""
    
    def __init__(self, data_dir: Path, split: str, processor: WhisperProcessor,
                 use_chunking: bool = False, chunk_strategy: str = "sequential"):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root data directory
            split: 'train', 'validation', or 'test'
            processor: Whisper processor
            use_chunking: Whether to use chunking for long audio (not needed if samples < 30s)
            chunk_strategy: Strategy for chunking if enabled
        """
        self.data_dir = data_dir
        self.audio_dir = data_dir / 'audio'
        self.processor = processor
        self.use_chunking = use_chunking
        self.chunk_strategy = chunk_strategy
        
        # Load metadata
        split_file = data_dir / 'splits' / f'{split}.csv'
        self.metadata = pd.read_csv(split_file)
        
        # Feature extractor settings
        self.feature_extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer
        self.sampling_rate = 16000
        
        # Initialize chunker only if needed
        if use_chunking:
            self.chunker = WhisperAudioChunker(overlap_seconds=2.0)
            self._create_chunked_dataset()
        else:
            # For datasets with samples already under 30s, no chunking needed
            self.samples = self.metadata
            print(f"Loaded {len(self.samples)} samples for {split} (no chunking needed)")
            
            # Check if any samples exceed 30 seconds
            if 'duration' in self.metadata.columns:
                long_samples = self.metadata[self.metadata['duration'] > 30]
                if len(long_samples) > 0:
                    print(f"Warning: {len(long_samples)} samples exceed 30 seconds!")
                    print("Consider using --use_chunking=True or regenerating dataset")
        
    def _create_chunked_dataset(self):
        """Pre-process dataset to create chunks from long audio files."""
        chunked_samples = []
        
        print(f"Creating chunked dataset with {self.chunk_strategy} strategy...")
        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            # For short audio (< 30s), just add as is
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
                # Create chunks for long audio
                num_chunks = int(np.ceil(row['duration'] / 28))  # 28s chunks with 2s overlap
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
        print(f"Created {len(self.samples)} samples from {len(self.metadata)} original files")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        row = self.samples.iloc[idx]
        
        # Load audio
        audio_path = self.audio_dir / row['audio_file']
        audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
        
        # Handle chunking if needed
        if self.use_chunking and row.get('is_chunked', False):
            # Extract chunk
            start_sample = int(row['chunk_start'] * self.sampling_rate)
            end_sample = int(row['chunk_end'] * self.sampling_rate)
            audio_chunk = audio[start_sample:end_sample]
            
            # Pad to 30 seconds if needed
            audio_chunk = self._pad_audio_to_30s(audio_chunk)
            
            # Align transcript - handle NaN values
            text_value = row['text'] if not pd.isna(row['text']) else ""
            total_duration = len(audio) / self.sampling_rate
            chunk_text = self.chunker._align_transcript(
                str(text_value), row['chunk_start'], row['chunk_end'], total_duration
            )
        else:
            # For samples already under 30s, just pad if needed
            audio_chunk = self._pad_audio_to_30s(audio)
            # Handle NaN values in text field
            chunk_text = row['text'] if not pd.isna(row['text']) else ""
        
        # Process audio
        input_features = self.feature_extractor(
            audio_chunk, 
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_features[0]
        
        # Process text (add special tokens for morse)
        # Handle empty text (noise-only samples)
        # Convert to string and handle NaN/None values
        if pd.isna(chunk_text) or chunk_text is None:
            chunk_text = ""
        else:
            chunk_text = str(chunk_text)
            
        if chunk_text.strip():
            text = f"<|morse|> {chunk_text} <|endoftext|>"
        else:
            # For noise-only samples, use a special token
            text = "<|noise|> <|endoftext|>"
        
        labels = self.tokenizer(text).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels,
            "text": chunk_text
        }
    
    def _pad_audio_to_30s(self, audio: np.ndarray) -> np.ndarray:
        """Pad audio to exactly 30 seconds."""
        target_length = 30 * self.sampling_rate  # 480,000 samples
        
        if len(audio) >= target_length:
            return audio[:target_length]
        
        # Pad with zeros
        pad_length = target_length - len(audio)
        return np.pad(audio, (0, pad_length), mode='constant', constant_values=0)


def prepare_dataset_memory_efficient(data_dir: Path, processor: WhisperProcessor, 
                                   use_chunking: bool = False,
                                   chunk_strategy: str = "sequential",
                                   max_samples: int = None) -> DatasetDict:
    """Prepare dataset for training - memory efficient version that processes audio on-demand."""
    datasets = {}
    
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
    
    def preprocess_function(examples):
        """Process audio files on-demand."""
        audio_files = examples["audio_file"]
        texts = examples["text"]
        
        # Process each sample
        input_features = []
        labels = []
        
        for audio_file, text in zip(audio_files, texts):
            # Load audio
            audio_path = str(audio_dir / audio_file)
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
        
        # Process in small batches to avoid memory issues
        batch_size = 32 if split == 'train' else 8
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
            desc=f"Processing {split} audio"
        )
        
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


def fine_tune_whisper(args):
    """Main fine-tuning function."""
    print(f"Fine-tuning Whisper model: {args.model_name}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # Prepare dataset
    print("Preparing dataset...")
    print(f"Chunking: {'enabled' if args.use_chunking else 'disabled'}")
    
    # Use memory-efficient dataset preparation
    dataset = prepare_dataset_memory_efficient(
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
    
    # Training arguments
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
    print("Starting training...")
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
        'validation_samples': len(dataset.get("validation", []))
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
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for morse code")
    
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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fine-tune
    fine_tune_whisper(args)


if __name__ == "__main__":
    main() 