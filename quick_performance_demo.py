"""
Quick demonstration of multiprocessing performance improvement
"""

import time
import tempfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from dataset_builder import MorseDatasetBuilder, create_default_config


def main():
    """Quick performance comparison."""
    print("=" * 60)
    print("Dataset Generation Performance Comparison")
    print("=" * 60)
    
    num_samples = 200  # Small number for quick demo
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with 1 worker (single-threaded)
        print("\n1. Single-threaded generation (1 worker):")
        builder1 = MorseDatasetBuilder(
            str(Path(temp_dir) / "single"),
            sample_rate=16000,
            max_duration_seconds=28.0,
            noise_sample_ratio=0.05
        )
        
        config = create_default_config()
        start = time.time()
        builder1.generate_dataset(num_samples, config, num_workers=1)
        single_time = time.time() - start
        
        print(f"\nTime taken: {single_time:.2f} seconds")
        print(f"Rate: {num_samples/single_time:.1f} samples/second")
        
        # Test with 4 workers (multi-threaded)
        print("\n2. Multi-threaded generation (4 workers):")
        builder2 = MorseDatasetBuilder(
            str(Path(temp_dir) / "multi"),
            sample_rate=16000,
            max_duration_seconds=28.0,
            noise_sample_ratio=0.05
        )
        
        start = time.time()
        builder2.generate_dataset(num_samples, config, num_workers=4)
        multi_time = time.time() - start
        
        print(f"\nTime taken: {multi_time:.2f} seconds")
        print(f"Rate: {num_samples/multi_time:.1f} samples/second")
        
        # Summary
        speedup = single_time / multi_time
        print("\n" + "=" * 60)
        print(f"Performance improvement: {speedup:.2f}x faster with 4 workers!")
        print(f"Time saved: {single_time - multi_time:.1f} seconds")
        print("=" * 60)


if __name__ == "__main__":
    main() 