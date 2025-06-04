"""
Benchmark script to test dataset generation performance with multiprocessing
"""

import time
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from dataset_builder import MorseDatasetBuilder, create_default_config


def benchmark_generation(num_samples: int, num_workers: int) -> float:
    """Benchmark dataset generation with specified number of workers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / f"benchmark_{num_workers}_workers"
        
        builder = MorseDatasetBuilder(
            str(output_dir),
            sample_rate=16000,
            max_duration_seconds=28.0,
            noise_sample_ratio=0.05
        )
        
        config = create_default_config()
        
        start_time = time.time()
        builder.generate_dataset(num_samples, config, num_workers=num_workers)
        end_time = time.time()
        
        return end_time - start_time


def main():
    """Run benchmarks with different worker configurations."""
    print("=" * 70)
    print("Dataset Generation Performance Benchmark")
    print("=" * 70)
    
    # Test configuration
    num_samples = 500  # Reasonable number for benchmark
    worker_configs = [1, 2, 4, 8, None]  # None = auto-detect CPU count
    
    print(f"\nGenerating {num_samples} samples with different worker configurations:\n")
    
    results = []
    
    for num_workers in worker_configs:
        worker_label = "auto" if num_workers is None else str(num_workers)
        print(f"Testing with {worker_label} worker(s)...", end='', flush=True)
        
        try:
            elapsed_time = benchmark_generation(num_samples, num_workers)
            samples_per_second = num_samples / elapsed_time
            
            results.append({
                'workers': worker_label,
                'time': elapsed_time,
                'samples_per_second': samples_per_second
            })
            
            print(f" Done! Time: {elapsed_time:.2f}s ({samples_per_second:.1f} samples/sec)")
            
        except Exception as e:
            print(f" Error: {e}")
    
    # Display results summary
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Samples/sec':<15} {'Speedup':<10}")
    print("-" * 57)
    
    if results:
        baseline_time = results[0]['time']  # Single worker as baseline
        
        for result in results:
            speedup = baseline_time / result['time']
            print(f"{result['workers']:<10} {result['time']:<12.2f} "
                  f"{result['samples_per_second']:<15.1f} {speedup:<10.2f}x")
    
    print("\n" + "=" * 70)
    print("Recommendations:")
    print("- For small datasets (<1000 samples): 2-4 workers")
    print("- For medium datasets (1000-10000 samples): 4-8 workers")
    print("- For large datasets (>10000 samples): auto (CPU count)")
    print("- Diminishing returns beyond 8 workers on most systems")
    print("=" * 70)


if __name__ == "__main__":
    main() 