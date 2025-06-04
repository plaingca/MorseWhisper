"""
Quick test to verify GPU optimization improvements.
This script runs a short training session to check GPU utilization patterns.
"""

import subprocess
import time
import GPUtil
import threading
import matplotlib.pyplot as plt
from pathlib import Path


def monitor_gpu(duration=60, interval=0.5):
    """Monitor GPU usage for a specified duration."""
    gpu_usage = []
    timestamps = []
    start_time = time.time()
    
    print("Monitoring GPU usage...")
    while time.time() - start_time < duration:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_usage.append(gpu.load * 100)
                timestamps.append(time.time() - start_time)
                print(f"\rGPU Usage: {gpu.load * 100:.1f}% | Memory: {gpu.memoryUsed:.0f}MB", end="")
        except:
            pass
        time.sleep(interval)
    
    print("\nMonitoring complete!")
    
    # Calculate statistics
    if gpu_usage:
        avg_usage = sum(gpu_usage) / len(gpu_usage)
        variance = sum((x - avg_usage) ** 2 for x in gpu_usage) / len(gpu_usage)
        print(f"\nStatistics:")
        print(f"  Average GPU Usage: {avg_usage:.1f}%")
        print(f"  Variance: {variance:.1f} (lower is more stable)")
        print(f"  Min/Max: {min(gpu_usage):.1f}% / {max(gpu_usage):.1f}%")
        
        # Plot GPU usage
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, gpu_usage, 'b-', linewidth=1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('GPU Utilization (%)')
        plt.title('GPU Utilization During Optimized Training')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig('gpu_usage_test.png', dpi=150)
        print(f"\nPlot saved to: gpu_usage_test.png")
        plt.close()


def main():
    print("GPU Optimization Test")
    print("=" * 50)
    
    # Check if training is already running
    try:
        gpus = GPUtil.getGPUs()
        if gpus and gpus[0].memoryUsed > 3000:
            print("Training appears to be running already!")
            print(f"Current GPU Memory Usage: {gpus[0].memoryUsed:.0f}MB")
            print(f"Current GPU Utilization: {gpus[0].load * 100:.1f}%")
            
            # Monitor for 60 seconds
            monitor_gpu(duration=60)
        else:
            print("No active training detected.")
            print("Please run the optimized training script in another terminal:")
            print("\npython training/fine_tune_optimized.py \\")
            print("    --data_dir experiments\\morse_whisper_20250603_115118\\data \\")
            print("    --output_dir test_optimization \\")
            print("    --batch_size 16 \\")
            print("    --dataloader_num_workers 4 \\")
            print("    --fp16 \\")
            print("    --max_steps 50")
            print("\nThen run this script again to monitor GPU usage.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 