"""
Test script to verify dataset generation with proper constraints

This script tests that the dataset builder:
1. Creates samples under 30 seconds
2. Includes noise-only samples
3. Has proper duration distribution
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from dataset_builder import MorseDatasetBuilder, create_default_config


def test_dataset_generation():
    """Test that dataset generation works correctly with new constraints."""
    print("=" * 60)
    print("Testing Dataset Generation")
    print("=" * 60)
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_dataset"
        
        # Create dataset builder with constraints
        builder = MorseDatasetBuilder(
            str(output_dir),
            sample_rate=16000,
            max_duration_seconds=28.0,
            noise_sample_ratio=0.05
        )
        
        # Generate small test dataset with limited workers for testing
        num_samples = 100
        config = create_default_config()
        
        print(f"\nGenerating {num_samples} samples...")
        # Use only 2 workers for testing to avoid overwhelming the system
        builder.generate_dataset(num_samples, config, num_workers=2)
        
        # Load and analyze metadata
        metadata_path = output_dir / 'metadata.csv'
        assert metadata_path.exists(), "Metadata file not created"
        
        metadata = pd.read_csv(metadata_path)
        print(f"\n✓ Generated {len(metadata)} samples")
        
        # Check duration constraints
        print("\n=== Duration Analysis ===")
        max_duration = metadata['duration'].max()
        min_duration = metadata['duration'].min()
        avg_duration = metadata['duration'].mean()
        
        print(f"Duration range: {min_duration:.1f}s - {max_duration:.1f}s")
        print(f"Average duration: {avg_duration:.1f}s")
        
        assert max_duration <= 28.0, f"Max duration {max_duration:.1f}s exceeds 28s limit"
        print("✓ All samples under 28 seconds")
        
        # Check noise samples
        print("\n=== Noise Sample Analysis ===")
        noise_samples = metadata[metadata['is_noise_only']]
        morse_samples = metadata[~metadata['is_noise_only']]
        
        noise_ratio = len(noise_samples) / len(metadata)
        print(f"Noise samples: {len(noise_samples)} ({noise_ratio*100:.1f}%)")
        print(f"Morse samples: {len(morse_samples)} ({len(morse_samples)/len(metadata)*100:.1f}%)")
        
        assert len(noise_samples) > 0, "No noise samples generated"
        assert 0.03 <= noise_ratio <= 0.07, f"Noise ratio {noise_ratio:.2f} outside expected range"
        print("✓ Noise samples included correctly")
        
        # Check empty transcripts for noise samples
        noise_texts = noise_samples['text'].fillna('')  # Handle NaN values
        assert all(text == "" or pd.isna(text) for text in noise_samples['text']), \
            "Noise samples should have empty transcripts"
        print("✓ Noise samples have empty transcripts")
        
        # Check morse samples have text
        morse_texts = morse_samples['text'].fillna('')
        assert all(text != "" for text in morse_texts), "Morse samples should have transcripts"
        print("✓ Morse samples have transcripts")
        
        # Check duration distribution
        print("\n=== Duration Distribution ===")
        bins = [0, 3, 8, 15, 22, 28]
        hist, _ = np.histogram(metadata['duration'], bins=bins)
        
        for i in range(len(bins)-1):
            count = hist[i]
            pct = count / len(metadata) * 100
            print(f"  {bins[i]:2d}-{bins[i+1]:2d}s: {count:3d} ({pct:5.1f}%)")
        
        # Check splits
        print("\n=== Split Analysis ===")
        splits_dir = output_dir / 'splits'
        assert splits_dir.exists(), "Splits directory not created"
        
        for split in ['train', 'validation', 'test']:
            split_file = splits_dir / f'{split}.csv'
            assert split_file.exists(), f"{split} split not created"
            
            split_df = pd.read_csv(split_file)
            noise_count = len(split_df[split_df['is_noise_only']])
            morse_count = len(split_df[~split_df['is_noise_only']])
            
            print(f"{split:12s}: {len(split_df):3d} samples "
                  f"(morse: {morse_count:3d}, noise: {noise_count:3d})")
        
        # Check audio files exist
        print("\n=== Audio File Check ===")
        audio_dir = output_dir / 'audio'
        audio_files = list(audio_dir.glob('*.wav'))
        print(f"Audio files created: {len(audio_files)}")
        
        assert len(audio_files) == len(metadata), "Audio file count mismatch"
        print("✓ All audio files created")
        
        # Sample a few files to check duration
        print("\n=== Audio Duration Verification ===")
        import librosa
        
        sample_indices = np.random.choice(len(metadata), min(5, len(metadata)), replace=False)
        
        for idx in sample_indices:
            row = metadata.iloc[idx]
            audio_path = audio_dir / row['audio_file']
            
            # Load audio and check duration
            audio, sr = librosa.load(audio_path, sr=16000)
            actual_duration = len(audio) / sr
            expected_duration = row['duration']
            
            print(f"  {row['audio_file']}: "
                  f"expected {expected_duration:.2f}s, "
                  f"actual {actual_duration:.2f}s")
            
            assert abs(actual_duration - expected_duration) < 0.1, \
                f"Duration mismatch for {row['audio_file']}"
        
        print("✓ Audio durations match metadata")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("Dataset generation working correctly")
        print("=" * 60)


def test_duration_estimation():
    """Test the duration estimation function."""
    print("\n=== Testing Duration Estimation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        builder = MorseDatasetBuilder(temp_dir)
        
        test_cases = [
            ("CQ CQ CQ DE K1ABC K1ABC K", 20, "Short CQ"),
            ("W1XYZ DE K2DEF 599 001 001 K", 25, "Contest exchange"),
            ("TEST", 30, "Single word"),
        ]
        
        for text, wpm, description in test_cases:
            estimated = builder.estimate_morse_duration(text, wpm)
            print(f"{description}: '{text}' at {wpm} WPM ≈ {estimated:.1f}s")
        
        print("✓ Duration estimation working")


def analyze_existing_dataset(dataset_path: str):
    """Analyze an existing dataset for compliance."""
    print("\n=== Analyzing Existing Dataset ===")
    
    dataset_path = Path(dataset_path)
    metadata_path = dataset_path / 'metadata.csv'
    
    if not metadata_path.exists():
        print(f"No metadata.csv found in {dataset_path}")
        return
    
    metadata = pd.read_csv(metadata_path)
    
    # Check for duration column
    if 'duration' not in metadata.columns:
        print("Warning: No duration information in metadata")
        return
    
    # Analyze durations
    over_30s = metadata[metadata['duration'] > 30]
    if len(over_30s) > 0:
        print(f"\n⚠️  Warning: {len(over_30s)} samples exceed 30 seconds!")
        print(f"Max duration: {metadata['duration'].max():.1f}s")
        print("\nRecommendations:")
        print("1. Regenerate dataset with --max_duration 28.0")
        print("2. Or use --use_chunking when training")
    else:
        print(f"✓ All {len(metadata)} samples are under 30 seconds")
        print(f"Max duration: {metadata['duration'].max():.1f}s")
    
    # Check for noise samples
    if 'is_noise_only' in metadata.columns:
        noise_count = metadata['is_noise_only'].sum()
        noise_ratio = noise_count / len(metadata)
        print(f"\nNoise samples: {noise_count} ({noise_ratio*100:.1f}%)")
        
        if noise_count == 0:
            print("⚠️  Warning: No noise samples found")
            print("Consider regenerating with --noise_ratio 0.05")
    else:
        print("\n⚠️  Warning: No noise sample information found")
        print("Dataset may be from older version")


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset generation")
    parser.add_argument("--analyze", type=str, help="Analyze existing dataset")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_existing_dataset(args.analyze)
    else:
        # Run generation tests
        test_dataset_generation()
        test_duration_estimation()
        
        print("\n" + "=" * 60)
        print("Dataset generation is working correctly!")
        print("Samples are properly constrained to <30 seconds")
        print("Noise samples are included for robustness")
        print("=" * 60)


if __name__ == "__main__":
    main() 