"""
Dataset Builder for Morse Code Fine-tuning

Generates a complete synthetic dataset with:
- Realistic morse code audio files (all under 30 seconds)
- Corresponding transcripts
- Various contest scenarios
- Multiple signal conditions
- Noise-only samples for robustness
- Proper train/validation/test splits
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import soundfile as sf
import yaml
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from morse_generator import MorseCodeGenerator
from callsign_generator import CallsignGenerator
from contest_exchanges import ContestExchangeGenerator
from noise_generator import NoiseGenerator


# Module-level function for multiprocessing
def generate_single_sample(args):
    """Generate a single sample (used for multiprocessing)."""
    idx, config, output_dir, sample_rate, max_duration, is_noise_only = args
    
    # Create generators for this process
    morse_gen = MorseCodeGenerator(sample_rate=sample_rate)
    callsign_gen = CallsignGenerator()
    exchange_gen = ContestExchangeGenerator()
    noise_gen = NoiseGenerator(sample_rate=sample_rate)
    
    # Create a temporary builder instance for this process
    builder = MorseDatasetBuilder(
        output_dir=output_dir,
        sample_rate=sample_rate,
        max_duration_seconds=max_duration,
        initialize_generators=False  # Don't initialize in __init__
    )
    
    # Set the generators
    builder.morse_gen = morse_gen
    builder.callsign_gen = callsign_gen
    builder.exchange_gen = exchange_gen
    builder.noise_gen = noise_gen
    
    try:
        audio_file, metadata = builder.generate_dataset_sample(idx, config, is_noise_only)
        return metadata
    except Exception as e:
        print(f"Error generating sample {idx}: {e}")
        return None


class MorseDatasetBuilder:
    """Build complete morse code dataset for Whisper fine-tuning."""
    
    def __init__(self, output_dir: str, sample_rate: int = 16000,
                 max_duration_seconds: float = 28.0,
                 noise_sample_ratio: float = 0.05,
                 initialize_generators: bool = True):
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Directory to save dataset
            sample_rate: Audio sample rate (16kHz for Whisper)
            max_duration_seconds: Maximum duration for samples (default 28s to stay under 30s)
            noise_sample_ratio: Ratio of noise-only samples to include
            initialize_generators: Whether to initialize generators (set False for multiprocessing)
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.noise_sample_ratio = noise_sample_ratio
        
        # Initialize generators only if requested (not in multiprocessing workers)
        if initialize_generators:
            self.morse_gen = MorseCodeGenerator(sample_rate=sample_rate)
            self.callsign_gen = CallsignGenerator()
            self.exchange_gen = ContestExchangeGenerator()
            self.noise_gen = NoiseGenerator(sample_rate=sample_rate)
        
        # Create output directories
        self.audio_dir = self.output_dir / 'audio'
        self.transcript_dir = self.output_dir / 'transcripts'
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Contest QSO templates
        self.qso_templates = {
            'CQ': [
                "CQ {contest} CQ {contest} DE {callsign} {callsign} K",
                "CQ CQ CQ DE {callsign} {callsign} {callsign}",
                "TEST {callsign} {callsign} TEST",
            ],
            'ANSWER': [
                "{dx_call} DE {my_call} {exchange} K",
                "{dx_call} {exchange} {my_call}",
                "{dx_call} {my_call} {exchange}",
            ],
            'ACKNOWLEDGMENT': [
                "TU {exchange} {callsign}",
                "R {exchange} TU",
                "{exchange} GL",
            ],
            'REPEAT': [
                "{callsign} AGN",
                "QRZ? DE {callsign}",
                "{partial}? {partial}?",
                "NR? NR?",
                "CALL?",
            ],
            'COMPLETION': [
                "TU QRZ? {callsign}",
                "QSL TU {contest} {callsign}",
                "73 GL {callsign} TEST",
            ]
        }
        
        # Duration distribution for samples under 30 seconds
        self.duration_weights = {
            (3, 8): 0.25,    # Very short messages (3-8s)
            (8, 15): 0.35,   # Short messages (8-15s)
            (15, 22): 0.30,  # Medium messages (15-22s)
            (22, 28): 0.10   # Long messages (22-28s) - staying under 30s
        }
        
    def generate_contest_qso(self, contest_type: str = 'CQWW', 
                           max_transmissions: int = 3) -> List[Dict]:
        """
        Generate a contest QSO with limited transmissions to fit in 30 seconds.
        
        Args:
            contest_type: Type of contest
            max_transmissions: Maximum number of transmissions to include
            
        Returns:
            List of dictionaries with 'text' and 'role' for each transmission
        """
        qso = []
        
        # Generate callsigns
        my_call, my_country = self.callsign_gen.generate_callsign()
        dx_call, dx_country = self.callsign_gen.generate_callsign()
        
        # Generate exchanges
        my_exchange = self.exchange_gen.generate_exchange(
            contest_type, {'country': my_country}
        )
        dx_exchange = self.exchange_gen.generate_exchange(
            contest_type, {'country': dx_country}
        )
        
        # Format exchanges
        my_exch_str = self.exchange_gen.format_exchange_string(my_exchange, contest_type)
        dx_exch_str = self.exchange_gen.format_exchange_string(dx_exchange, contest_type)
        
        # Choose a subset of the QSO to fit time constraints
        qso_type = random.choice(['cq_only', 'exchange', 'full_short'])
        
        if qso_type == 'cq_only':
            # Just a CQ call
            cq_template = random.choice(self.qso_templates['CQ'])
            cq_text = cq_template.format(
                contest=contest_type if random.random() < 0.3 else 'TEST',
                callsign=my_call
            )
            qso.append({'text': cq_text, 'role': 'CQ', 'callsign': my_call})
            
        elif qso_type == 'exchange':
            # Just an exchange
            answer_template = random.choice(self.qso_templates['ANSWER'])
            answer_text = answer_template.format(
                dx_call=my_call,
                my_call=dx_call,
                exchange=dx_exch_str
            )
            qso.append({'text': answer_text, 'role': 'ANSWER', 'callsign': dx_call})
            
        else:  # full_short
            # Short version of full QSO
            # CQ
            cq_text = f"CQ DE {my_call} K"
            qso.append({'text': cq_text, 'role': 'CQ', 'callsign': my_call})
            
            # Answer
            answer_text = f"{my_call} DE {dx_call} {dx_exch_str}"
            qso.append({'text': answer_text, 'role': 'ANSWER', 'callsign': dx_call})
            
            # Ack (if we have room)
            if max_transmissions > 2:
                ack_text = f"TU {my_exch_str}"
                qso.append({'text': ack_text, 'role': 'ACK', 'callsign': my_call})
        
        return qso[:max_transmissions]
    
    def generate_single_transmission(self, contest_type: str = 'CQWW') -> Dict:
        """Generate a single contest transmission."""
        # Generate callsign and exchange
        callsign, country = self.callsign_gen.generate_callsign()
        exchange = self.exchange_gen.generate_exchange(contest_type, {'country': country})
        exchange_str = self.exchange_gen.format_exchange_string(exchange, contest_type)
        
        # Choose transmission type
        tx_type = random.choice(['CQ', 'EXCHANGE', 'PARTIAL', 'SHORT'])
        
        if tx_type == 'CQ':
            template = random.choice(self.qso_templates['CQ'])
            text = template.format(contest='TEST', callsign=callsign)
        elif tx_type == 'EXCHANGE':
            text = f"{callsign} {exchange_str}"
        elif tx_type == 'PARTIAL':
            # Simulate partial copy
            if random.random() < 0.5:
                # Partial callsign
                text = callsign[:random.randint(2, len(callsign)-1)] + "?"
            else:
                # Missing part of exchange
                text = f"{callsign} {exchange_str.split()[0]} ?"
        else:  # SHORT
            # Very short transmissions
            text = random.choice([
                callsign,
                "QRZ?",
                "AGN",
                "TU",
                f"{callsign} TEST",
                "73",
                exchange_str
            ])
        
        return {'text': text, 'callsign': callsign, 'type': tx_type}
    
    def generate_noise_only_sample(self, duration: float, condition: str) -> np.ndarray:
        """
        Generate a noise-only sample (no morse signal).
        
        Args:
            duration: Duration in seconds
            condition: Noise condition preset
            
        Returns:
            Audio array with only noise
        """
        # Create silence
        samples = int(duration * self.sample_rate)
        audio = np.zeros(samples)
        
        # Apply noise conditions
        audio = self.noise_gen.apply_realistic_conditions(audio, condition)
        
        return audio
    
    def estimate_morse_duration(self, text: str, wpm: int) -> float:
        """
        Estimate the duration of morse code text at given WPM.
        
        Args:
            text: Text to transmit
            wpm: Words per minute
            
        Returns:
            Estimated duration in seconds
        """
        # Basic estimation: average word is 5 characters
        # Each character takes about (60 / (wpm * 5)) seconds
        # Add spacing between words
        words = text.split()
        char_count = sum(len(word) for word in words)
        word_count = len(words)
        
        # Time per character (including inter-character spacing)
        char_time = 60.0 / (wpm * 5)  
        
        # Add word spacing (7 units vs 3 for character)
        word_space_time = char_time * 4 * (word_count - 1)
        
        estimated_duration = (char_count * char_time * 1.2) + word_space_time
        
        return estimated_duration
    
    def generate_audio_within_duration(self, text: str, target_duration: float,
                                     wpm: int, condition_preset: str) -> Tuple[np.ndarray, str]:
        """
        Generate audio that fits within the target duration.
        
        Args:
            text: Base text to transmit
            target_duration: Target duration in seconds
            wpm: Words per minute
            condition_preset: Noise condition
            
        Returns:
            Tuple of (audio, actual_text)
        """
        # Estimate duration
        estimated_duration = self.estimate_morse_duration(text, wpm)
        
        # If text is too long, truncate it
        if estimated_duration > target_duration * 0.9:
            words = text.split()
            while len(words) > 1 and self.estimate_morse_duration(' '.join(words), wpm) > target_duration * 0.8:
                words.pop()
            text = ' '.join(words)
        
        # Generate morse audio
        audio = self.morse_gen.text_to_audio(
            text,
            wpm=wpm,
            timing_variation=random.uniform(2, 8)
        )
        
        # Apply conditions
        audio = self.noise_gen.apply_realistic_conditions(audio, condition_preset)
        
        # Add padding at start and end
        pad_before = int(random.uniform(0.5, 1.5) * self.sample_rate)
        pad_after = int(random.uniform(0.5, 1.5) * self.sample_rate)
        
        # Ensure total duration doesn't exceed target
        total_duration = (len(audio) + pad_before + pad_after) / self.sample_rate
        if total_duration > target_duration:
            # Reduce padding
            excess = total_duration - target_duration
            excess_samples = int(excess * self.sample_rate)
            if excess_samples < pad_after:
                pad_after -= excess_samples
            else:
                pad_after = int(0.2 * self.sample_rate)
                pad_before = max(int(0.2 * self.sample_rate), pad_before - (excess_samples - pad_after))
        
        audio = np.concatenate([
            np.zeros(pad_before),
            audio,
            np.zeros(pad_after)
        ])
        
        # Final check - truncate if still too long
        max_samples = int(target_duration * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        return audio, text
    
    def add_prosigns_and_abbreviations(self, text: str) -> str:
        """Add common prosigns and abbreviations."""
        # Common substitutions
        substitutions = {
            'AND': 'ES',
            'ARE': 'R',
            'YOU': 'U',
            'YOUR': 'UR',
            'SIGNAL': 'SIG',
            'REPORT': 'RPT',
            'OVER': 'K',
            'ROGER': 'R',
            'THANKS': 'TU',
            'THANK YOU': 'TU',
        }
        
        # Random abbreviations
        if random.random() < 0.3:
            for full, abbr in substitutions.items():
                if full in text.upper():
                    text = text.upper().replace(full, abbr)
        
        return text
    
    def get_target_duration(self) -> float:
        """Get target duration for next audio sample based on distribution."""
        # Sample from duration distribution
        ranges = list(self.duration_weights.keys())
        weights = list(self.duration_weights.values())
        
        selected_range = random.choices(ranges, weights=weights)[0]
        return random.uniform(selected_range[0], selected_range[1])
    
    def generate_dataset_sample(self, idx: int, config: Dict, 
                               is_noise_only: bool = False) -> Tuple[str, Dict]:
        """
        Generate a single dataset sample.
        
        Args:
            idx: Sample index
            config: Configuration dictionary
            is_noise_only: Whether to generate noise-only sample
            
        Returns:
            Tuple of (audio_filename, metadata_dict)
        """
        # Get target duration (always under 30 seconds)
        target_duration = self.get_target_duration()
        
        # Choose condition
        condition = random.choice(config['conditions'])
        
        if is_noise_only:
            # Generate noise-only sample
            audio = self.generate_noise_only_sample(target_duration, condition)
            text = ""  # Empty transcript for noise
            sample_type = 'noise_only'
            wpm = 0
            contest_type = 'NOISE'
        else:
            # Choose contest type
            contest_type = random.choice(config['contest_types'])
            
            # Generate appropriate length content
            if target_duration < 10:
                # Short single transmission
                tx = self.generate_single_transmission(contest_type)
                base_text = tx['text']
                sample_type = tx['type'].lower()
            elif target_duration < 20:
                # Medium - partial QSO
                qso = self.generate_contest_qso(contest_type, max_transmissions=2)
                base_text = " ".join([tx['text'] for tx in qso])
                sample_type = 'partial_qso'
            else:
                # Longer - full QSO or multiple transmissions
                if random.random() < 0.6:
                    qso = self.generate_contest_qso(contest_type, max_transmissions=3)
                    base_text = " ".join([tx['text'] for tx in qso])
                    sample_type = 'qso'
                else:
                    # Multiple single transmissions
                    transmissions = []
                    for _ in range(random.randint(2, 4)):
                        tx = self.generate_single_transmission(contest_type)
                        transmissions.append(tx['text'])
                    base_text = " ".join(transmissions)
                    sample_type = 'multi_tx'
            
            # Add prosigns/abbreviations
            base_text = self.add_prosigns_and_abbreviations(base_text)
            
            # Generate audio parameters
            wpm = random.randint(config['wpm_range'][0], config['wpm_range'][1])
            
            # Generate audio that fits within duration
            audio, text = self.generate_audio_within_duration(
                base_text, target_duration, wpm, condition
            )
        
        # Ensure audio is under 30 seconds
        max_samples = int(self.max_duration_seconds * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        actual_duration = len(audio) / self.sample_rate
        
        # Save audio
        audio_filename = f"morse_{idx:06d}.wav"
        audio_path = self.audio_dir / audio_filename
        sf.write(audio_path, audio, self.sample_rate)
        
        # Create metadata
        metadata = {
            'id': idx,
            'audio_file': audio_filename,
            'text': text.upper() if text else "",  # Empty for noise-only
            'contest_type': contest_type,
            'sample_type': sample_type,
            'wpm': wpm,
            'condition': condition,
            'duration': actual_duration,
            'is_noise_only': is_noise_only
        }
        
        return audio_filename, metadata
    
    def generate_dataset(self, num_samples: int, config: Dict, num_workers: Optional[int] = None):
        """Generate complete dataset with noise samples using multiprocessing."""
        print(f"Generating {num_samples} morse code samples...")
        print(f"All samples will be under {self.max_duration_seconds} seconds")
        print(f"Including {self.noise_sample_ratio*100:.0f}% noise-only samples")
        
        # Determine number of workers
        if num_workers is None:
            num_workers = min(cpu_count(), 8)  # Cap at 8 workers
        print(f"Using {num_workers} worker processes")
        
        # Calculate number of noise samples
        num_noise_samples = int(num_samples * self.noise_sample_ratio)
        num_morse_samples = num_samples - num_noise_samples
        
        # Prepare arguments for parallel processing
        morse_args = [
            (idx, config, str(self.output_dir), self.sample_rate, 
             self.max_duration_seconds, False)
            for idx in range(num_morse_samples)
        ]
        
        noise_args = [
            (num_morse_samples + idx, config, str(self.output_dir), 
             self.sample_rate, self.max_duration_seconds, True)
            for idx in range(num_noise_samples)
        ]
        
        all_args = morse_args + noise_args
        
        # Generate samples in parallel
        print(f"\nGenerating samples using {num_workers} processes...")
        all_metadata = []
        
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered for better progress tracking
            with tqdm(total=len(all_args)) as pbar:
                for result in pool.imap_unordered(generate_single_sample, all_args):
                    if result is not None:
                        all_metadata.append(result)
                    pbar.update()
        
        # Sort by ID to maintain order
        all_metadata.sort(key=lambda x: x['id'])
        
        # Save metadata
        metadata_df = pd.DataFrame(all_metadata)
        # Ensure empty text is saved as empty string, not NaN
        metadata_df['text'] = metadata_df['text'].fillna('')
        metadata_df.to_csv(self.output_dir / 'metadata.csv', index=False)
        
        # Print statistics
        print("\nDataset statistics:")
        print(f"Total samples: {len(all_metadata)}")
        morse_count = len([m for m in all_metadata if not m['is_noise_only']])
        noise_count = len([m for m in all_metadata if m['is_noise_only']])
        print(f"Morse samples: {morse_count}")
        print(f"Noise-only samples: {noise_count}")
        print(f"Average duration: {metadata_df['duration'].mean():.1f}s")
        print(f"Max duration: {metadata_df['duration'].max():.1f}s")
        print(f"Min duration: {metadata_df['duration'].min():.1f}s")
        
        # Duration distribution
        print("\nDuration distribution:")
        bins = [0, 5, 10, 15, 20, 25, 30]
        hist, _ = np.histogram(metadata_df['duration'], bins=bins)
        for i in range(len(bins)-1):
            count = hist[i]
            pct = count / len(metadata_df) * 100
            print(f"  {bins[i]:2d}-{bins[i+1]:2d}s: {count:4d} ({pct:5.1f}%)")
        
        # Create splits
        self.create_splits(metadata_df, config.get('split_ratios', [0.8, 0.1, 0.1]))
        
        # Save dataset info
        dataset_info = {
            'num_samples': len(all_metadata),
            'num_morse_samples': morse_count,
            'num_noise_samples': noise_count,
            'sample_rate': self.sample_rate,
            'max_duration': self.max_duration_seconds,
            'contest_types': config['contest_types'],
            'wpm_range': config['wpm_range'],
            'conditions': config['conditions'],
            'avg_duration': float(metadata_df['duration'].mean()),
            'num_workers': num_workers,
            'creation_date': pd.Timestamp.now().isoformat()
        }
        
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nDataset generation complete!")
        
    def create_splits(self, metadata_df: pd.DataFrame, 
                     split_ratios: List[float] = [0.8, 0.1, 0.1]):
        """Create train/validation/test splits with balanced noise samples."""
        n_samples = len(metadata_df)
        
        # Separate morse and noise samples
        morse_df = metadata_df[~metadata_df['is_noise_only']]
        noise_df = metadata_df[metadata_df['is_noise_only']]
        
        # Shuffle
        morse_indices = np.random.permutation(len(morse_df))
        noise_indices = np.random.permutation(len(noise_df))
        
        # Calculate split sizes for each type
        morse_train_size = int(len(morse_indices) * split_ratios[0])
        morse_val_size = int(len(morse_indices) * split_ratios[1])
        
        noise_train_size = int(len(noise_indices) * split_ratios[0])
        noise_val_size = int(len(noise_indices) * split_ratios[1])
        
        # Create splits
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Add morse samples
        train_indices.extend(morse_df.index[morse_indices[:morse_train_size]])
        val_indices.extend(morse_df.index[morse_indices[morse_train_size:morse_train_size + morse_val_size]])
        test_indices.extend(morse_df.index[morse_indices[morse_train_size + morse_val_size:]])
        
        # Add noise samples
        train_indices.extend(noise_df.index[noise_indices[:noise_train_size]])
        val_indices.extend(noise_df.index[noise_indices[noise_train_size:noise_train_size + noise_val_size]])
        test_indices.extend(noise_df.index[noise_indices[noise_train_size + noise_val_size:]])
        
        # Shuffle within each split
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        # Save split files
        splits = {
            'train': metadata_df.loc[train_indices],
            'validation': metadata_df.loc[val_indices],
            'test': metadata_df.loc[test_indices]
        }
        
        splits_dir = self.output_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        for split_name, split_df in splits.items():
            # Ensure empty text is saved as empty string in splits too
            split_df = split_df.copy()
            split_df['text'] = split_df['text'].fillna('')
            split_df.to_csv(splits_dir / f'{split_name}.csv', index=False)
            
            # Create transcript files for Whisper fine-tuning
            transcript_file = splits_dir / f'{split_name}_transcripts.txt'
            with open(transcript_file, 'w') as f:
                for _, row in split_df.iterrows():
                    # Ensure text is not NaN when writing transcripts
                    text = row['text'] if pd.notna(row['text']) else ""
                    f.write(f"{row['audio_file']}|{text}\n")
            
            # Print statistics
            morse_count = len(split_df[~split_df['is_noise_only']])
            noise_count = len(split_df[split_df['is_noise_only']])
            print(f"\n{split_name.capitalize()} split: {len(split_df)} samples")
            print(f"  Morse: {morse_count} ({morse_count/len(split_df)*100:.1f}%)")
            print(f"  Noise: {noise_count} ({noise_count/len(split_df)*100:.1f}%)")
            print(f"  Avg duration: {split_df['duration'].mean():.1f}s")


def create_default_config():
    """Create default configuration."""
    config = {
        'contest_types': ['CQWW', 'CQWPX', 'ARRLDX', 'FIELD_DAY', 'NAQP'],
        'wpm_range': [15, 40],
        'conditions': ['contest_good', 'contest_moderate', 'contest_poor'],
        'split_ratios': [0.8, 0.1, 0.1]
    }
    return config


def main():
    parser = argparse.ArgumentParser(description='Generate morse code dataset')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file (YAML)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Audio sample rate (16000 for Whisper)')
    parser.add_argument('--max_duration', type=float, default=28.0,
                       help='Maximum audio duration in seconds (default 28 to stay under 30)')
    parser.add_argument('--noise_ratio', type=float, default=0.05,
                       help='Ratio of noise-only samples (default 5%%)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (default: number of CPU cores)')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()
    
    # Create dataset builder
    builder = MorseDatasetBuilder(
        args.output_dir, 
        args.sample_rate,
        max_duration_seconds=args.max_duration,
        noise_sample_ratio=args.noise_ratio
    )
    
    # Generate dataset with multiprocessing
    builder.generate_dataset(args.num_samples, config, num_workers=args.num_workers)


if __name__ == "__main__":
    main() 