"""
MorseWhisper Inference

Use the fine-tuned Whisper model to decode morse code audio files.
"""

import argparse
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import List, Optional
import json
from tqdm import tqdm

from transformers import WhisperProcessor, WhisperForConditionalGeneration


class MorseWhisperInference:
    """Inference class for morse code decoding with fine-tuned Whisper."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to fine-tuned model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_path = Path(model_path)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def decode_audio(self, audio_path: str, 
                    num_beams: int = 5,
                    temperature: float = 0.0,
                    language: Optional[str] = None) -> dict:
        """
        Decode morse code from audio file.
        
        Args:
            audio_path: Path to audio file
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling (0 = greedy)
            language: Language hint (not used for morse)
            
        Returns:
            Dictionary with transcription and metadata
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
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
                num_beams=num_beams,
                temperature=temperature
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        # Clean up morse prefix if present
        if transcription.startswith("<|morse|>"):
            transcription = transcription[9:].strip()
        
        return {
            'audio_file': str(audio_path),
            'transcription': transcription,
            'duration': duration,
            'num_beams': num_beams,
            'temperature': temperature
        }
    
    def decode_batch(self, audio_files: List[str], 
                    batch_size: int = 8,
                    **decode_kwargs) -> List[dict]:
        """
        Decode multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            batch_size: Batch size for processing
            **decode_kwargs: Additional arguments for decode_audio
            
        Returns:
            List of transcription dictionaries
        """
        results = []
        
        print(f"Decoding {len(audio_files)} audio files...")
        
        for audio_path in tqdm(audio_files):
            try:
                result = self.decode_audio(audio_path, **decode_kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'audio_file': str(audio_path),
                    'transcription': '',
                    'error': str(e)
                })
        
        return results
    
    def decode_real_time(self, audio_stream):
        """
        Decode morse code from real-time audio stream.
        
        Note: This is a placeholder for real-time functionality.
        Implement with pyaudio or similar for actual real-time decoding.
        """
        raise NotImplementedError("Real-time decoding not yet implemented")


def process_audio_file(model_path: str, audio_path: str, 
                      output_format: str = 'text',
                      verbose: bool = False) -> str:
    """
    Simple function to decode a single audio file.
    
    Args:
        model_path: Path to fine-tuned model
        audio_path: Path to audio file
        output_format: 'text' or 'json'
        verbose: Print additional information
        
    Returns:
        Transcription string or JSON
    """
    # Initialize inference
    inference = MorseWhisperInference(model_path)
    
    # Decode audio
    result = inference.decode_audio(audio_path)
    
    if verbose:
        print(f"\nAudio: {audio_path}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Transcription: {result['transcription']}")
    
    if output_format == 'json':
        return json.dumps(result, indent=2)
    else:
        return result['transcription']


def main():
    parser = argparse.ArgumentParser(
        description="Decode morse code using fine-tuned Whisper model"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to fine-tuned model directory"
    )
    
    # Input arguments
    parser.add_argument(
        "--audio_file",
        type=str,
        help="Single audio file to decode"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        help="Directory of audio files to decode"
    )
    parser.add_argument(
        "--file_list",
        type=str,
        help="Text file with list of audio paths"
    )
    
    # Decoding arguments
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 = greedy)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=['text', 'json', 'csv'],
        default='text',
        help="Output format"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = MorseWhisperInference(args.model_path)
    
    # Collect audio files
    audio_files = []
    
    if args.audio_file:
        audio_files = [args.audio_file]
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
        audio_files = [str(f) for f in audio_files]
    elif args.file_list:
        with open(args.file_list, 'r') as f:
            audio_files = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Must specify --audio_file, --audio_dir, or --file_list")
    
    # Process audio files
    if len(audio_files) == 1:
        # Single file
        result = inference.decode_audio(
            audio_files[0],
            num_beams=args.num_beams,
            temperature=args.temperature
        )
        results = [result]
    else:
        # Multiple files
        results = inference.decode_batch(
            audio_files,
            num_beams=args.num_beams,
            temperature=args.temperature
        )
    
    # Format output
    if args.output_format == 'text':
        output_lines = []
        for result in results:
            if 'error' in result:
                output_lines.append(f"{result['audio_file']}: ERROR - {result['error']}")
            else:
                output_lines.append(f"{result['audio_file']}: {result['transcription']}")
        output_text = '\n'.join(output_lines)
    
    elif args.output_format == 'json':
        output_text = json.dumps(results, indent=2)
    
    else:  # csv
        import csv
        import io
        output = io.StringIO()
        writer = csv.DictWriter(
            output, 
            fieldnames=['audio_file', 'transcription', 'duration', 'error']
        )
        writer.writeheader()
        for result in results:
            writer.writerow({
                'audio_file': result['audio_file'],
                'transcription': result.get('transcription', ''),
                'duration': result.get('duration', ''),
                'error': result.get('error', '')
            })
        output_text = output.getvalue()
    
    # Save or print output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Results saved to {args.output}")
    else:
        print(output_text)
    
    # Print summary if verbose
    if args.verbose:
        print(f"\n{'='*60}")
        print(f"Processed {len(results)} audio files")
        successful = len([r for r in results if 'error' not in r])
        print(f"Successful: {successful}")
        print(f"Errors: {len(results) - successful}")


if __name__ == "__main__":
    main() 