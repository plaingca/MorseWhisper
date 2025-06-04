"""
Evaluate Fine-tuned Whisper Model on Morse Code

This script evaluates the performance of a fine-tuned Whisper model
on morse code recognition, computing WER and providing detailed analysis.
"""

import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio
import evaluate
import librosa
from jiwer import wer, cer, compute_measures

# Import the audio chunking module
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from audio_chunking import WhisperAudioChunker


class MorseWhisperEvaluator:
    """Evaluate Whisper model on morse code recognition with chunking support."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to fine-tuned model
            device: Device to use for inference
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        print(f"Loading model from {model_path}...")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load chunking configuration if available
        chunk_config_path = self.model_path / "chunk_config.json"
        if chunk_config_path.exists():
            with open(chunk_config_path, 'r') as f:
                self.chunk_config = json.load(f)
                print(f"Loaded chunk config: {self.chunk_config}")
        else:
            self.chunk_config = {'chunk_strategy': 'sequential', 'overlap_seconds': 2.0}
            print("No chunk config found, using defaults")
        
        # Initialize chunker
        self.chunker = WhisperAudioChunker(
            overlap_seconds=self.chunk_config.get('overlap_seconds', 2.0)
        )
        
        # Metrics
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        
    def transcribe_audio(self, audio_path: str, use_chunking: bool = True) -> str:
        """
        Transcribe a single audio file with chunking support.
        
        Args:
            audio_path: Path to audio file
            use_chunking: Whether to use chunking for long audio
            
        Returns:
            Transcribed text
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
        # Handle chunking for long audio
        if use_chunking and duration > 30:
            return self._transcribe_with_chunks(audio)
        else:
            # Process short audio directly
            # Pad to 30 seconds if needed
            if len(audio) < self.chunker.CHUNK_LENGTH_SAMPLES:
                audio = self.chunker.pad_audio(audio)
            
            # Process audio
            input_features = self.processor(
                audio, 
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=225,
                    num_beams=5,
                    temperature=0.0
                )
            
            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # Remove morse prefix if present
            if transcription.startswith("<|morse|>"):
                transcription = transcription[9:].strip()
            
            return transcription
    
    def _transcribe_with_chunks(self, audio: np.ndarray) -> str:
        """
        Transcribe long audio using chunking.
        
        Args:
            audio: Audio array at 16kHz
            
        Returns:
            Merged transcription
        """
        # Create chunks
        chunks = self.chunker.chunk_audio(audio)
        
        # Process each chunk
        chunk_predictions = []
        for chunk_info in chunks:
            chunk_audio = chunk_info['audio']
            
            # Process chunk
            input_features = self.processor(
                chunk_audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=225,
                    num_beams=5,
                    temperature=0.0
                )
            
            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            # Remove morse prefix if present
            if transcription.startswith("<|morse|>"):
                transcription = transcription[9:].strip()
            
            chunk_predictions.append({
                'text': transcription,
                'start_time': chunk_info['start_time'],
                'end_time': chunk_info['end_time']
            })
        
        # Merge predictions
        merged_text = self.chunker.merge_chunk_predictions(chunk_predictions)
        
        return merged_text
    
    def evaluate_dataset(self, data_dir: Path, split: str = "test",
                        use_chunking: bool = True) -> Dict:
        """
        Evaluate on a dataset split.
        
        Args:
            data_dir: Dataset directory
            split: Split to evaluate ('test', 'validation', etc.)
            use_chunking: Whether to use chunking for long audio
            
        Returns:
            Dictionary with evaluation results
        """
        # Load metadata
        split_file = data_dir / 'splits' / f'{split}.csv'
        metadata = pd.read_csv(split_file)
        
        print(f"Evaluating on {len(metadata)} samples from {split} set...")
        print(f"Chunking enabled: {use_chunking}")
        
        predictions = []
        references = []
        detailed_results = []
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
            audio_path = data_dir / 'audio' / row['audio_file']
            
            # Get prediction
            try:
                prediction = self.transcribe_audio(str(audio_path), use_chunking=use_chunking)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                prediction = ""
            
            reference = row['text']
            
            # Handle NaN/empty references (noise-only samples)
            if pd.isna(reference) or reference is None:
                reference = ""
            else:
                reference = str(reference)
            
            predictions.append(prediction)
            references.append(reference)
            
            # Compute sample-level metrics
            # Skip WER/CER calculation for empty references (noise samples)
            if reference.strip():
                sample_wer = wer(reference, prediction)
                sample_cer = cer(reference, prediction)
            else:
                # For noise samples, check if prediction is also empty
                sample_wer = 0.0 if not prediction.strip() else 1.0
                sample_cer = 0.0 if not prediction.strip() else 1.0
            
            detailed_results.append({
                'id': row.get('id', idx),
                'audio_file': row['audio_file'],
                'reference': reference,
                'prediction': prediction,
                'wer': sample_wer,
                'cer': sample_cer,
                'contest_type': row.get('contest_type', 'unknown'),
                'wpm': row.get('wpm', 0),
                'condition': row.get('condition', 'unknown'),
                'duration': row.get('duration', 0),
                'used_chunking': use_chunking and row.get('duration', 0) > 30,
                'is_noise_only': row.get('is_noise_only', False)
            })
        
        # Compute overall metrics - filter out noise samples
        morse_predictions = []
        morse_references = []
        
        for pred, ref in zip(predictions, references):
            if ref.strip():  # Only include non-empty references
                morse_predictions.append(pred)
                morse_references.append(ref)
        
        if morse_predictions:
            overall_wer = self.wer_metric.compute(
                predictions=morse_predictions, 
                references=morse_references
            )
            overall_cer = self.cer_metric.compute(
                predictions=morse_predictions, 
                references=morse_references
            )
        else:
            overall_wer = 0.0
            overall_cer = 0.0
        
        # Calculate noise accuracy separately
        noise_results = [r for r in detailed_results if r.get('is_noise_only', False)]
        noise_accuracy = sum(1 for r in noise_results if r['wer'] == 0.0) / len(noise_results) if noise_results else 1.0
        
        # Analyze errors
        error_analysis = self.analyze_errors(detailed_results)
        
        # Analyze chunking performance
        chunking_analysis = self.analyze_chunking_performance(detailed_results)
        
        results = {
            'overall_wer': overall_wer,
            'overall_cer': overall_cer,
            'noise_accuracy': noise_accuracy,
            'num_samples': len(metadata),
            'num_morse_samples': len(morse_predictions),
            'num_noise_samples': len(noise_results),
            'detailed_results': detailed_results,
            'error_analysis': error_analysis,
            'chunking_analysis': chunking_analysis
        }
        
        return results
    
    def analyze_chunking_performance(self, detailed_results: List[Dict]) -> Dict:
        """Analyze performance difference between chunked and non-chunked audio."""
        chunked_results = [r for r in detailed_results if r.get('used_chunking', False)]
        non_chunked_results = [r for r in detailed_results if not r.get('used_chunking', False)]
        
        analysis = {
            'chunked': {
                'count': len(chunked_results),
                'avg_wer': np.mean([r['wer'] for r in chunked_results]) if chunked_results else 0,
                'avg_duration': np.mean([r['duration'] for r in chunked_results]) if chunked_results else 0
            },
            'non_chunked': {
                'count': len(non_chunked_results),
                'avg_wer': np.mean([r['wer'] for r in non_chunked_results]) if non_chunked_results else 0,
                'avg_duration': np.mean([r['duration'] for r in non_chunked_results]) if non_chunked_results else 0
            }
        }
        
        # Analyze by duration bins
        duration_bins = [(0, 30), (30, 60), (60, 90), (90, float('inf'))]
        analysis['by_duration'] = {}
        
        for min_dur, max_dur in duration_bins:
            bin_results = [r for r in detailed_results 
                          if min_dur <= r['duration'] < max_dur]
            if bin_results:
                bin_name = f"{min_dur}-{max_dur if max_dur != float('inf') else '∞'}s"
                analysis['by_duration'][bin_name] = {
                    'count': len(bin_results),
                    'avg_wer': np.mean([r['wer'] for r in bin_results])
                }
        
        return analysis
    
    def analyze_errors(self, detailed_results: List[Dict]) -> Dict:
        """Analyze error patterns."""
        analysis = {
            'by_contest_type': defaultdict(list),
            'by_wpm': defaultdict(list),
            'by_condition': defaultdict(list),
            'by_duration': defaultdict(list),
            'callsign_errors': [],
            'number_errors': [],
            'common_substitutions': defaultdict(int)
        }
        
        for result in detailed_results:
            # Group by categories
            contest_type = result['contest_type']
            condition = result['condition']
            wpm_bucket = (result['wpm'] // 5) * 5  # 5 WPM buckets
            duration_bucket = (int(result['duration']) // 30) * 30  # 30s buckets
            
            analysis['by_contest_type'][contest_type].append(result['wer'])
            analysis['by_condition'][condition].append(result['wer'])
            analysis['by_wpm'][f'{wpm_bucket}-{wpm_bucket+4}'].append(result['wer'])
            analysis['by_duration'][f'{duration_bucket}-{duration_bucket+29}s'].append(result['wer'])
            
            # Analyze specific error types
            ref_tokens = result['reference'].split()
            pred_tokens = result['prediction'].split()
            
            # Check for callsign errors (tokens with numbers and letters)
            for ref_token, pred_token in zip(ref_tokens, pred_tokens[:len(ref_tokens)]):
                if any(c.isdigit() for c in ref_token) and any(c.isalpha() for c in ref_token):
                    if ref_token != pred_token:
                        analysis['callsign_errors'].append({
                            'reference': ref_token,
                            'prediction': pred_token
                        })
                
                # Check for number errors
                if ref_token.isdigit() and ref_token != pred_token:
                    analysis['number_errors'].append({
                        'reference': ref_token,
                        'prediction': pred_token
                    })
                
                # Track substitutions
                if ref_token != pred_token:
                    sub_key = f"{ref_token} → {pred_token}"
                    analysis['common_substitutions'][sub_key] += 1
        
        # Compute average WER by category
        for category in ['by_contest_type', 'by_wpm', 'by_condition', 'by_duration']:
            for key, wer_list in analysis[category].items():
                analysis[category][key] = {
                    'avg_wer': np.mean(wer_list) if wer_list else 0,
                    'count': len(wer_list)
                }
        
        # Sort common substitutions
        analysis['common_substitutions'] = dict(
            sorted(analysis['common_substitutions'].items(), 
                   key=lambda x: x[1], reverse=True)[:20]
        )
        
        return analysis
    
    def plot_results(self, results: Dict, output_dir: Path):
        """Generate visualization plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Extract data for plotting
        detailed_results = pd.DataFrame(results['detailed_results'])
        
        # 1. WER distribution
        plt.figure(figsize=(10, 6))
        plt.hist(detailed_results['wer'], bins=50, edgecolor='black')
        plt.xlabel('Word Error Rate')
        plt.ylabel('Count')
        plt.title(f'WER Distribution (Overall: {results["overall_wer"]:.2%})')
        plt.axvline(results['overall_wer'], color='red', linestyle='--', 
                   label=f'Mean: {results["overall_wer"]:.2%}')
        plt.legend()
        plt.savefig(output_dir / 'wer_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. WER by contest type
        plt.figure(figsize=(12, 6))
        contest_wer = detailed_results.groupby('contest_type')['wer'].agg(['mean', 'std', 'count'])
        contest_wer['mean'].plot(kind='bar', yerr=contest_wer['std'], capsize=5)
        plt.xlabel('Contest Type')
        plt.ylabel('Average WER')
        plt.title('WER by Contest Type')
        plt.xticks(rotation=45)
        for i, (idx, row) in enumerate(contest_wer.iterrows()):
            plt.text(i, row['mean'] + row['std'] + 0.01, f'n={row["count"]}', 
                    ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(output_dir / 'wer_by_contest.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. WER by duration (new plot)
        plt.figure(figsize=(12, 6))
        detailed_results['duration_bucket'] = (detailed_results['duration'] // 30) * 30
        duration_wer = detailed_results.groupby('duration_bucket')['wer'].agg(['mean', 'std', 'count'])
        duration_wer['mean'].plot(kind='line', marker='o', markersize=8)
        plt.fill_between(duration_wer.index, 
                        duration_wer['mean'] - duration_wer['std'], 
                        duration_wer['mean'] + duration_wer['std'], 
                        alpha=0.3)
        plt.xlabel('Audio Duration (seconds)')
        plt.ylabel('Average WER')
        plt.title('WER by Audio Duration')
        plt.grid(True, alpha=0.3)
        
        # Add vertical line at 30 seconds
        plt.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='30s chunk boundary')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'wer_by_duration.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Chunking performance comparison (if applicable)
        if 'chunking_analysis' in results:
            chunking_data = results['chunking_analysis']
            if chunking_data['chunked']['count'] > 0 and chunking_data['non_chunked']['count'] > 0:
                plt.figure(figsize=(10, 6))
                
                categories = ['Non-chunked\n(<30s)', 'Chunked\n(>30s)']
                wer_values = [chunking_data['non_chunked']['avg_wer'], 
                             chunking_data['chunked']['avg_wer']]
                counts = [chunking_data['non_chunked']['count'], 
                         chunking_data['chunked']['count']]
                
                bars = plt.bar(categories, wer_values)
                plt.ylabel('Average WER')
                plt.title('WER Comparison: Chunked vs Non-chunked Audio')
                
                # Add count labels
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'n={count}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'chunking_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Continue with other plots...
        # 5. WER by WPM
        plt.figure(figsize=(12, 6))
        detailed_results['wpm_bucket'] = (detailed_results['wpm'] // 5) * 5
        wpm_wer = detailed_results.groupby('wpm_bucket')['wer'].agg(['mean', 'std', 'count'])
        wpm_wer['mean'].plot(kind='line', marker='o', markersize=8)
        plt.fill_between(wpm_wer.index, 
                        wpm_wer['mean'] - wpm_wer['std'], 
                        wpm_wer['mean'] + wpm_wer['std'], 
                        alpha=0.3)
        plt.xlabel('WPM')
        plt.ylabel('Average WER')
        plt.title('WER by Speed (WPM)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'wer_by_wpm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. WER by condition
        plt.figure(figsize=(10, 6))
        condition_wer = detailed_results.groupby('condition')['wer'].agg(['mean', 'std', 'count'])
        condition_wer['mean'].plot(kind='bar', yerr=condition_wer['std'], capsize=5)
        plt.xlabel('Signal Condition')
        plt.ylabel('Average WER')
        plt.title('WER by Signal Condition')
        plt.xticks(rotation=45)
        for i, (idx, row) in enumerate(condition_wer.iterrows()):
            plt.text(i, row['mean'] + row['std'] + 0.01, f'n={row["count"]}', 
                    ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(output_dir / 'wer_by_condition.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Common errors
        error_analysis = results['error_analysis']
        if error_analysis['common_substitutions']:
            plt.figure(figsize=(12, 8))
            subs = error_analysis['common_substitutions']
            items = list(subs.items())[:15]  # Top 15
            labels = [item[0] for item in items]
            values = [item[1] for item in items]
            
            plt.barh(labels, values)
            plt.xlabel('Count')
            plt.title('Most Common Substitution Errors')
            plt.tight_layout()
            plt.savefig(output_dir / 'common_errors.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {output_dir}")
    
    def generate_report(self, results: Dict, output_path: Path):
        """Generate detailed evaluation report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Morse Code Whisper Evaluation Report\n\n")
            
            # Overall metrics
            f.write("## Overall Performance\n")
            f.write(f"- **Word Error Rate (WER)**: {results['overall_wer']:.2%}\n")
            f.write(f"- **Character Error Rate (CER)**: {results['overall_cer']:.2%}\n")
            f.write(f"- **Noise Sample Accuracy**: {results.get('noise_accuracy', 0):.2%}\n")
            f.write(f"- **Number of samples**: {results['num_samples']}\n")
            f.write(f"  - Morse samples: {results.get('num_morse_samples', 0)}\n")
            f.write(f"  - Noise samples: {results.get('num_noise_samples', 0)}\n\n")
            
            # Chunking analysis
            if 'chunking_analysis' in results:
                f.write("## Chunking Analysis\n")
                chunking = results['chunking_analysis']
                f.write(f"- **Non-chunked audio (<30s)**: {chunking['non_chunked']['count']} samples, "
                       f"avg WER: {chunking['non_chunked']['avg_wer']:.2%}\n")
                f.write(f"- **Chunked audio (>30s)**: {chunking['chunked']['count']} samples, "
                       f"avg WER: {chunking['chunked']['avg_wer']:.2%}\n\n")
                
                f.write("### Performance by Duration\n")
                for duration_range, stats in chunking['by_duration'].items():
                    f.write(f"- **{duration_range}**: {stats['avg_wer']:.2%} (n={stats['count']})\n")
                f.write("\n")
            
            # Performance by category
            error_analysis = results['error_analysis']
            
            f.write("## Performance by Contest Type\n")
            for contest, stats in error_analysis['by_contest_type'].items():
                f.write(f"- **{contest}**: {stats['avg_wer']:.2%} (n={stats['count']})\n")
            
            f.write("\n## Performance by Signal Condition\n")
            for condition, stats in error_analysis['by_condition'].items():
                f.write(f"- **{condition}**: {stats['avg_wer']:.2%} (n={stats['count']})\n")
            
            f.write("\n## Performance by Speed (WPM)\n")
            for wpm_range, stats in sorted(error_analysis['by_wpm'].items()):
                f.write(f"- **{wpm_range} WPM**: {stats['avg_wer']:.2%} (n={stats['count']})\n")
            
            # Common errors
            f.write("\n## Most Common Errors\n")
            f.write("### Substitutions (Top 20)\n")
            for sub, count in list(error_analysis['common_substitutions'].items())[:20]:
                f.write(f"- `{sub}`: {count} times\n")
            
            # Callsign errors
            if error_analysis['callsign_errors']:
                f.write("\n### Sample Callsign Errors\n")
                for i, error in enumerate(error_analysis['callsign_errors'][:10]):
                    f.write(f"- `{error['reference']}` → `{error['prediction']}`\n")
            
            # Perfect transcriptions
            perfect = [r for r in results['detailed_results'] if r['wer'] == 0]
            f.write(f"\n## Perfect Transcriptions: {len(perfect)}/{results['num_samples']} ({len(perfect)/results['num_samples']:.1%})\n")
            
            # Worst performers
            f.write("\n## Worst Performing Samples\n")
            worst = sorted(results['detailed_results'], key=lambda x: x['wer'], reverse=True)[:10]
            for i, sample in enumerate(worst, 1):
                f.write(f"\n### Sample {i} (WER: {sample['wer']:.2%})\n")
                f.write(f"- **Reference**: `{sample['reference']}`\n")
                f.write(f"- **Prediction**: `{sample['prediction']}`\n")
                f.write(f"- **Conditions**: {sample['wpm']} WPM, {sample['condition']}, "
                       f"duration: {sample['duration']:.1f}s\n")
                if sample.get('used_chunking'):
                    f.write(f"- **Note**: Audio was chunked (>30s)\n")
        
        print(f"Report saved to {output_path}")


def compare_with_baseline(evaluator: MorseWhisperEvaluator, 
                         data_dir: Path, 
                         baseline_model: str = "openai/whisper-small") -> Dict:
    """Compare fine-tuned model with baseline Whisper."""
    print("\nEvaluating baseline model for comparison...")
    
    # Create baseline evaluator
    baseline_eval = MorseWhisperEvaluator(baseline_model)
    
    # Evaluate on same test set
    baseline_results = baseline_eval.evaluate_dataset(data_dir, "test")
    
    # Comparison
    comparison = {
        'fine_tuned_wer': evaluator.results['overall_wer'],
        'baseline_wer': baseline_results['overall_wer'],
        'improvement': (baseline_results['overall_wer'] - evaluator.results['overall_wer']) / baseline_results['overall_wer'],
        'fine_tuned_cer': evaluator.results['overall_cer'],
        'baseline_cer': baseline_results['overall_cer']
    }
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper on morse code")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to fine-tuned model")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save results")
    parser.add_argument("--no_chunking", action="store_true",
                       help="Disable chunking for long audio")
    parser.add_argument("--compare_baseline", action="store_true",
                       help="Compare with baseline Whisper model")
    parser.add_argument("--baseline_model", type=str, default="openai/whisper-small",
                       help="Baseline model to compare against")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = MorseWhisperEvaluator(args.model_path)
    
    # Evaluate
    print(f"Evaluating on {args.split} split...")
    use_chunking = not args.no_chunking
    results = evaluator.evaluate_dataset(Path(args.data_dir), args.split, use_chunking=use_chunking)
    evaluator.results = results  # Store for comparison
    
    # Save raw results
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        # Convert numpy types for JSON serialization
        json_results = {
            'overall_wer': float(results['overall_wer']),
            'overall_cer': float(results['overall_cer']),
            'noise_accuracy': float(results['noise_accuracy']),
            'num_samples': results['num_samples'],
            'num_morse_samples': results['num_morse_samples'],
            'num_noise_samples': results['num_noise_samples'],
            'error_analysis': results['error_analysis'],
            'chunking_analysis': results['chunking_analysis']
        }
        json.dump(json_results, f, indent=2)
    
    # Save detailed results
    detailed_df = pd.DataFrame(results['detailed_results'])
    detailed_df.to_csv(output_dir / 'detailed_results.csv', index=False)
    
    # Generate plots
    evaluator.plot_results(results, output_dir)
    
    # Generate report
    evaluator.generate_report(results, output_dir / 'evaluation_report.md')
    
    # Compare with baseline if requested
    if args.compare_baseline:
        comparison = compare_with_baseline(
            evaluator, 
            Path(args.data_dir), 
            args.baseline_model
        )
        
        print("\n=== Model Comparison ===")
        print(f"Fine-tuned WER: {comparison['fine_tuned_wer']:.2%}")
        print(f"Baseline WER: {comparison['baseline_wer']:.2%}")
        print(f"Improvement: {comparison['improvement']:.1%}")
        
        with open(output_dir / 'comparison.json', 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Overall WER: {results['overall_wer']:.2%}")
    print(f"Overall CER: {results['overall_cer']:.2%}")
    print(f"Noise Accuracy: {results.get('noise_accuracy', 0):.2%}")
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"  - Morse samples: {results.get('num_morse_samples', 0)}")
    print(f"  - Noise samples: {results.get('num_noise_samples', 0)}")
    
    if 'chunking_analysis' in results:
        chunking = results['chunking_analysis']
        print(f"\nChunking statistics:")
        print(f"- Non-chunked (<30s): {chunking['non_chunked']['count']} samples, "
              f"WER: {chunking['non_chunked']['avg_wer']:.2%}")
        print(f"- Chunked (>30s): {chunking['chunked']['count']} samples, "
              f"WER: {chunking['chunked']['avg_wer']:.2%}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main() 