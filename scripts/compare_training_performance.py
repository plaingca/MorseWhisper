"""
Compare training performance between original and optimized fine-tuning scripts.

This script helps measure GPU utilization, training speed, and memory usage.
"""

import subprocess
import time
import threading
import psutil
import GPUtil
from pathlib import Path
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime


class GPUMonitor:
    """Monitor GPU utilization during training."""
    
    def __init__(self, interval=1.0):
        self.interval = interval
        self.gpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        self.monitoring = False
        self.thread = None
        
    def start(self):
        """Start monitoring GPU."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        """Stop monitoring GPU."""
        self.monitoring = False
        if self.thread:
            self.thread.join()
            
    def _monitor(self):
        """Monitor GPU in background thread."""
        while self.monitoring:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    self.gpu_usage.append(gpu.load * 100)
                    self.memory_usage.append(gpu.memoryUsed)
                    self.timestamps.append(time.time())
            except:
                pass
            time.sleep(self.interval)
            
    def get_stats(self):
        """Get GPU statistics."""
        if not self.gpu_usage:
            return {}
            
        return {
            'avg_gpu_usage': sum(self.gpu_usage) / len(self.gpu_usage),
            'max_gpu_usage': max(self.gpu_usage),
            'min_gpu_usage': min(self.gpu_usage),
            'avg_memory_mb': sum(self.memory_usage) / len(self.memory_usage),
            'max_memory_mb': max(self.memory_usage),
            'gpu_usage_variance': self._calculate_variance(self.gpu_usage)
        }
        
    def _calculate_variance(self, data):
        """Calculate variance to measure burstiness."""
        if len(data) < 2:
            return 0
        mean = sum(data) / len(data)
        return sum((x - mean) ** 2 for x in data) / len(data)
        
    def plot_usage(self, title="GPU Usage"):
        """Plot GPU usage over time."""
        if not self.timestamps:
            return
            
        # Convert timestamps to relative time
        start_time = self.timestamps[0]
        relative_times = [(t - start_time) for t in self.timestamps]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # GPU utilization
        ax1.plot(relative_times, self.gpu_usage, 'b-', linewidth=1)
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Memory usage
        ax2.plot(relative_times, self.memory_usage, 'r-', linewidth=1)
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_xlabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def run_training_test(script_path, args, duration_seconds=300):
    """Run a training test for a specified duration."""
    print(f"\nRunning {script_path} for {duration_seconds} seconds...")
    
    # Start GPU monitoring
    monitor = GPUMonitor(interval=0.5)
    monitor.start()
    
    # Build command
    cmd = ['python', script_path] + args + ['--max_steps', '100']  # Limit steps for testing
    
    # Record start time
    start_time = time.time()
    
    # Run training
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for specified duration or process completion
        process.wait(timeout=duration_seconds)
        
    except subprocess.TimeoutExpired:
        # Normal case - we stop after duration
        process.terminate()
        process.wait()
        
    # Record end time
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # Stop monitoring
    monitor.stop()
    
    # Get GPU stats
    stats = monitor.get_stats()
    stats['duration'] = actual_duration
    
    return monitor, stats


def main():
    parser = argparse.ArgumentParser(description="Compare training performance")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--duration", type=int, default=300, 
                       help="Test duration in seconds (default: 300)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size to test")
    parser.add_argument("--output_dir", default="./performance_comparison",
                       help="Output directory for results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Common training arguments
    common_args = [
        '--data_dir', args.data_dir,
        '--batch_size', str(args.batch_size),
        '--logging_steps', '10',
        '--fp16'  # Use FP16 for both
    ]
    
    results = {}
    
    # Test original implementation
    print("\n" + "="*60)
    print("Testing ORIGINAL implementation...")
    print("="*60)
    
    original_args = common_args + [
        '--output_dir', str(output_dir / 'original_test')
    ]
    
    original_monitor, original_stats = run_training_test(
        'training/fine_tune.py',
        original_args,
        args.duration
    )
    
    results['original'] = original_stats
    
    # Save original plot
    fig = original_monitor.plot_usage("Original Implementation - GPU Usage")
    fig.savefig(output_dir / 'gpu_usage_original.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test optimized implementation
    print("\n" + "="*60)
    print("Testing OPTIMIZED implementation...")
    print("="*60)
    
    optimized_args = common_args + [
        '--output_dir', str(output_dir / 'optimized_test'),
        '--dataloader_num_workers', '4',
        '--bf16'  # Use BF16 for RTX 4080
    ]
    
    optimized_monitor, optimized_stats = run_training_test(
        'training/fine_tune_optimized.py',
        optimized_args,
        args.duration
    )
    
    results['optimized'] = optimized_stats
    
    # Save optimized plot
    fig = optimized_monitor.plot_usage("Optimized Implementation - GPU Usage")
    fig.savefig(output_dir / 'gpu_usage_optimized.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"\nAverage GPU Utilization:")
    print(f"  Original:  {results['original']['avg_gpu_usage']:.1f}%")
    print(f"  Optimized: {results['optimized']['avg_gpu_usage']:.1f}%")
    improvement = (results['optimized']['avg_gpu_usage'] / results['original']['avg_gpu_usage'] - 1) * 100
    print(f"  Improvement: {improvement:+.1f}%")
    
    print(f"\nGPU Usage Variance (lower = more stable):")
    print(f"  Original:  {results['original']['gpu_usage_variance']:.1f}")
    print(f"  Optimized: {results['optimized']['gpu_usage_variance']:.1f}")
    
    print(f"\nPeak Memory Usage:")
    print(f"  Original:  {results['original']['max_memory_mb']:.0f} MB")
    print(f"  Optimized: {results['optimized']['max_memory_mb']:.0f} MB")
    
    # Save results
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot both GPU usage patterns on same graph
    if original_monitor.timestamps and optimized_monitor.timestamps:
        # Original
        orig_start = original_monitor.timestamps[0]
        orig_times = [(t - orig_start) for t in original_monitor.timestamps]
        ax.plot(orig_times, original_monitor.gpu_usage, 'r-', 
               label='Original', alpha=0.7, linewidth=1)
        
        # Optimized
        opt_start = optimized_monitor.timestamps[0]
        opt_times = [(t - opt_start) for t in optimized_monitor.timestamps]
        ax.plot(opt_times, optimized_monitor.gpu_usage, 'b-', 
               label='Optimized', alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Utilization Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'gpu_usage_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nResults saved to: {output_dir}")
    print("  - gpu_usage_original.png")
    print("  - gpu_usage_optimized.png")
    print("  - gpu_usage_comparison.png")
    print("  - comparison_results.json")


if __name__ == "__main__":
    main() 