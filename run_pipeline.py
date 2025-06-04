"""
MorseWhisper Pipeline Runner

Complete pipeline for fine-tuning Whisper on morse code:
1. Generate synthetic dataset
2. Fine-tune Whisper
3. Evaluate performance
4. Compare with baseline
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import yaml


class MorsePipeline:
    """Orchestrate the complete morse code fine-tuning pipeline."""
    
    def __init__(self, config_path: str, output_base: str = "experiments"):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
            output_base: Base directory for experiment outputs
        """
        self.config_path = Path(config_path)
        self.output_base = Path(output_base)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_base / f"morse_whisper_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up paths
        self.data_dir = self.experiment_dir / "data"
        self.model_dir = self.experiment_dir / "model"
        self.eval_dir = self.experiment_dir / "evaluation"
        self.log_dir = self.experiment_dir / "logs"
        
        # Create directories
        for dir_path in [self.data_dir, self.model_dir, self.eval_dir, self.log_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Copy config to experiment directory
        import shutil
        shutil.copy(self.config_path, self.experiment_dir / "config.yaml")
        
    def run_command(self, cmd: list, log_name: str) -> int:
        """Run a command and log output."""
        log_file = self.log_dir / f"{log_name}.log"
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"Log: {log_file}")
        print(f"{'='*60}\n")
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            
        return process.returncode
    
    def generate_dataset(self, num_samples: int = 10000):
        """Generate synthetic morse code dataset."""
        print("\n🎵 Generating Morse Code Dataset...")
        
        cmd = [
            sys.executable,
            "src/dataset_builder.py",
            "--num_samples", str(num_samples),
            "--output_dir", str(self.data_dir),
            "--config", str(self.config_path),
            "--sample_rate", "16000"
        ]
        
        return_code = self.run_command(cmd, "dataset_generation")
        
        if return_code != 0:
            raise RuntimeError("Dataset generation failed!")
        
        # Verify dataset
        metadata_path = self.data_dir / "metadata.csv"
        if metadata_path.exists():
            import pandas as pd
            df = pd.read_csv(metadata_path)
            print(f"\n✅ Dataset generated: {len(df)} samples")
            print(f"   Audio files: {self.data_dir / 'audio'}")
            print(f"   Splits: {self.data_dir / 'splits'}")
        else:
            raise RuntimeError("Dataset metadata not found!")
    
    def fine_tune_model(self, model_name: str = "openai/whisper-small",
                       batch_size: int = 16,
                       num_epochs: int = 3,
                       learning_rate: float = 1e-5,
                       fp16: bool = True):
        """Fine-tune Whisper model."""
        print("\n🤖 Fine-tuning Whisper Model...")
        
        cmd = [
            sys.executable,
            "training/fine_tune.py",
            "--model_name", model_name,
            "--data_dir", str(self.data_dir),
            "--output_dir", str(self.model_dir),
            "--batch_size", str(batch_size),
            "--num_epochs", str(num_epochs),
            "--learning_rate", str(learning_rate),
            "--warmup_steps", "500",
            "--save_steps", "500",
            "--eval_steps", "500",
            "--logging_steps", "100"
        ]
        
        if fp16:
            cmd.append("--fp16")
        
        return_code = self.run_command(cmd, "fine_tuning")
        
        if return_code != 0:
            raise RuntimeError("Model fine-tuning failed!")
        
        print(f"\n✅ Model fine-tuned and saved to: {self.model_dir}")
    
    def evaluate_model(self, compare_baseline: bool = True):
        """Evaluate fine-tuned model."""
        print("\n📊 Evaluating Model Performance...")
        
        cmd = [
            sys.executable,
            "training/evaluate_model.py",
            "--model_path", str(self.model_dir),
            "--data_dir", str(self.data_dir),
            "--output_dir", str(self.eval_dir)
        ]
        
        if compare_baseline:
            cmd.append("--compare_baseline")
        
        return_code = self.run_command(cmd, "evaluation")
        
        if return_code != 0:
            raise RuntimeError("Model evaluation failed!")
        
        # Load and display results
        results_path = self.eval_dir / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print(f"\n✅ Evaluation Complete:")
            print(f"   Overall WER: {results['overall_wer']:.2%}")
            print(f"   Overall CER: {results['overall_cer']:.2%}")
            print(f"   Report: {self.eval_dir / 'evaluation_report.md'}")
            print(f"   Plots: {self.eval_dir}")
        
        # Display comparison if available
        comparison_path = self.eval_dir / "comparison.json"
        if comparison_path.exists():
            with open(comparison_path, 'r') as f:
                comparison = json.load(f)
            
            print(f"\n📈 Model Comparison:")
            print(f"   Baseline WER: {comparison['baseline_wer']:.2%}")
            print(f"   Fine-tuned WER: {comparison['fine_tuned_wer']:.2%}")
            print(f"   Improvement: {comparison['improvement']:.1%}")
    
    def run_full_pipeline(self, num_samples: int = 10000,
                         model_name: str = "openai/whisper-small",
                         batch_size: int = 16,
                         num_epochs: int = 3,
                         skip_generation: bool = False,
                         skip_training: bool = False):
        """Run complete pipeline."""
        start_time = time.time()
        
        print(f"\n🚀 Starting MorseWhisper Pipeline")
        print(f"   Experiment: {self.experiment_dir}")
        print(f"   Config: {self.config_path}")
        
        try:
            # Step 1: Generate dataset
            if not skip_generation:
                self.generate_dataset(num_samples)
            else:
                print("\n⏭️  Skipping dataset generation")
            
            # Step 2: Fine-tune model
            if not skip_training:
                self.fine_tune_model(
                    model_name=model_name,
                    batch_size=batch_size,
                    num_epochs=num_epochs
                )
            else:
                print("\n⏭️  Skipping model training")
            
            # Step 3: Evaluate model
            self.evaluate_model(compare_baseline=True)
            
            # Summary
            elapsed_time = time.time() - start_time
            print(f"\n🎉 Pipeline Complete!")
            print(f"   Total time: {elapsed_time/60:.1f} minutes")
            print(f"   Results: {self.experiment_dir}")
            
            # Save pipeline summary
            summary = {
                "experiment_dir": str(self.experiment_dir),
                "config": str(self.config_path),
                "num_samples": num_samples,
                "model_name": model_name,
                "elapsed_time_minutes": elapsed_time / 60,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(self.experiment_dir / "pipeline_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Run MorseWhisper pipeline")
    
    # Pipeline arguments
    parser.add_argument("--config", type=str, default="configs/contest_cw.yaml",
                       help="Configuration file path")
    parser.add_argument("--output_base", type=str, default="experiments",
                       help="Base directory for experiment outputs")
    
    # Dataset arguments
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of samples to generate")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip dataset generation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="openai/whisper-small",
                       help="Whisper model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip model training")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = MorsePipeline(args.config, args.output_base)
    pipeline.run_full_pipeline(
        num_samples=args.num_samples,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        skip_generation=args.skip_generation,
        skip_training=args.skip_training
    )


if __name__ == "__main__":
    main() 