"""
Audio Chunking Utilities for Whisper

This module provides utilities for chunking and padding audio to fit
Whisper's 30-second context window requirement.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import torch


class WhisperAudioChunker:
    """Handle audio chunking and padding for Whisper's 30-second window."""
    
    # Whisper constants
    SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
    CHUNK_LENGTH_SECONDS = 30  # Whisper processes 30-second chunks
    CHUNK_LENGTH_SAMPLES = CHUNK_LENGTH_SECONDS * SAMPLE_RATE  # 480,000 samples
    N_FFT = 400  # FFT window size
    HOP_LENGTH = 160  # Hop length for STFT
    N_FRAMES = 3000  # Number of mel spectrogram frames for 30 seconds
    
    def __init__(self, overlap_seconds: float = 2.0, 
                 min_chunk_seconds: float = 1.0):
        """
        Initialize the chunker.
        
        Args:
            overlap_seconds: Overlap between chunks in seconds
            min_chunk_seconds: Minimum chunk duration to process
        """
        self.overlap_seconds = overlap_seconds
        self.overlap_samples = int(overlap_seconds * self.SAMPLE_RATE)
        self.min_chunk_samples = int(min_chunk_seconds * self.SAMPLE_RATE)
        
    def chunk_audio(self, audio: np.ndarray, 
                   transcript: Optional[str] = None) -> List[Dict]:
        """
        Chunk audio into 30-second segments with optional overlap.
        
        Args:
            audio: Audio array at 16kHz
            transcript: Optional transcript to align with chunks
            
        Returns:
            List of dictionaries containing:
                - 'audio': Audio chunk (padded to 30s if needed)
                - 'start_time': Start time in seconds
                - 'end_time': End time in seconds
                - 'transcript': Aligned transcript (if provided)
                - 'is_padded': Whether chunk was padded
        """
        audio_length = len(audio)
        chunks = []
        
        # If audio is shorter than 30 seconds, just pad it
        if audio_length <= self.CHUNK_LENGTH_SAMPLES:
            padded_audio = self.pad_audio(audio)
            chunks.append({
                'audio': padded_audio,
                'start_time': 0.0,
                'end_time': audio_length / self.SAMPLE_RATE,
                'transcript': transcript,
                'is_padded': True,
                'original_length': audio_length
            })
            return chunks
        
        # Calculate stride (how much to move forward for each chunk)
        stride = self.CHUNK_LENGTH_SAMPLES - self.overlap_samples
        
        # Create chunks
        start_sample = 0
        while start_sample < audio_length:
            end_sample = min(start_sample + self.CHUNK_LENGTH_SAMPLES, audio_length)
            
            # Extract chunk
            chunk = audio[start_sample:end_sample]
            
            # Skip if chunk is too short (unless it's the last chunk)
            if len(chunk) < self.min_chunk_samples and end_sample < audio_length:
                break
            
            # Pad if necessary
            if len(chunk) < self.CHUNK_LENGTH_SAMPLES:
                chunk = self.pad_audio(chunk)
                is_padded = True
            else:
                is_padded = False
            
            # Calculate times
            start_time = start_sample / self.SAMPLE_RATE
            end_time = end_sample / self.SAMPLE_RATE
            
            # Align transcript if provided
            chunk_transcript = None
            if transcript:
                chunk_transcript = self._align_transcript(
                    transcript, start_time, end_time, audio_length / self.SAMPLE_RATE
                )
            
            chunks.append({
                'audio': chunk,
                'start_time': start_time,
                'end_time': end_time,
                'transcript': chunk_transcript,
                'is_padded': is_padded,
                'original_length': end_sample - start_sample
            })
            
            # Move to next chunk
            start_sample += stride
            
            # If we've covered the entire audio, break
            if start_sample >= audio_length - self.min_chunk_samples:
                break
        
        return chunks
    
    def pad_audio(self, audio: np.ndarray, 
                  pad_mode: str = 'constant') -> np.ndarray:
        """
        Pad audio to 30 seconds.
        
        Args:
            audio: Audio array
            pad_mode: Padding mode ('constant', 'edge', 'reflect', etc.)
            
        Returns:
            Padded audio of exactly 30 seconds
        """
        if len(audio) >= self.CHUNK_LENGTH_SAMPLES:
            return audio[:self.CHUNK_LENGTH_SAMPLES]
        
        pad_length = self.CHUNK_LENGTH_SAMPLES - len(audio)
        
        if pad_mode == 'constant':
            # Pad with zeros
            padded = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
        else:
            # Use numpy's other padding modes
            padded = np.pad(audio, (0, pad_length), mode=pad_mode)
        
        return padded
    
    def _align_transcript(self, transcript: str, start_time: float, 
                         end_time: float, total_duration: float) -> str:
        """
        Align transcript with audio chunk (simple word-based alignment).
        
        This is a simple implementation that distributes words evenly
        across the audio duration. For morse code, this should work
        reasonably well since transmission is relatively uniform.
        
        Args:
            transcript: Full transcript
            start_time: Chunk start time in seconds
            end_time: Chunk end time in seconds
            total_duration: Total audio duration in seconds
            
        Returns:
            Aligned transcript for the chunk
        """
        words = transcript.split()
        if not words:
            return ""
        
        # Calculate word timing (assuming uniform distribution)
        words_per_second = len(words) / total_duration
        
        # Estimate word indices for this chunk
        start_word_idx = int(start_time * words_per_second)
        end_word_idx = int(end_time * words_per_second) + 1
        
        # Ensure indices are within bounds
        start_word_idx = max(0, min(start_word_idx, len(words) - 1))
        end_word_idx = max(start_word_idx + 1, min(end_word_idx, len(words)))
        
        # Extract words for this chunk
        chunk_words = words[start_word_idx:end_word_idx]
        
        return " ".join(chunk_words)
    
    def merge_chunk_predictions(self, predictions: List[Dict], 
                               remove_duplicates: bool = True) -> str:
        """
        Merge predictions from overlapping chunks.
        
        Args:
            predictions: List of prediction dictionaries with 'text' and timing info
            remove_duplicates: Whether to remove duplicate words at boundaries
            
        Returns:
            Merged transcript
        """
        if not predictions:
            return ""
        
        if len(predictions) == 1:
            return predictions[0]['text']
        
        # Sort by start time
        sorted_preds = sorted(predictions, key=lambda x: x['start_time'])
        
        merged_words = []
        last_words = []
        
        for i, pred in enumerate(sorted_preds):
            words = pred['text'].split()
            
            if i == 0:
                merged_words.extend(words)
                last_words = words[-3:] if len(words) > 3 else words
            else:
                # Check for overlap
                if remove_duplicates and last_words:
                    # Find overlap between end of previous and start of current
                    overlap_found = False
                    for j in range(min(len(last_words), len(words))):
                        if last_words[-j-1:] == words[:j+1]:
                            # Skip the overlapping part
                            merged_words.extend(words[j+1:])
                            overlap_found = True
                            break
                    
                    if not overlap_found:
                        merged_words.extend(words)
                else:
                    merged_words.extend(words)
                
                last_words = words[-3:] if len(words) > 3 else words
        
        return " ".join(merged_words)
    
    def prepare_batch_chunks(self, chunks: List[Dict], 
                           feature_extractor) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of chunks for Whisper processing.
        
        Args:
            chunks: List of chunk dictionaries
            feature_extractor: Whisper feature extractor
            
        Returns:
            Batch dictionary ready for model input
        """
        # Extract audio arrays
        audio_arrays = [chunk['audio'] for chunk in chunks]
        
        # Process with feature extractor
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        return inputs


def create_training_chunks(audio: np.ndarray, transcript: str,
                         chunk_strategy: str = "sequential") -> List[Dict]:
    """
    Create training chunks with different strategies.
    
    Args:
        audio: Audio array at 16kHz
        transcript: Full transcript
        chunk_strategy: Strategy for chunking
            - "sequential": Non-overlapping sequential chunks
            - "sliding": Overlapping sliding window
            - "random": Random chunks from the audio
            
    Returns:
        List of training chunks
    """
    chunker = WhisperAudioChunker()
    
    if chunk_strategy == "sequential":
        # Non-overlapping chunks - set overlap to 0
        chunker.overlap_seconds = 0
        chunker.overlap_samples = 0
        return chunker.chunk_audio(audio, transcript)
    
    elif chunk_strategy == "sliding":
        # Overlapping chunks (default 2 second overlap)
        return chunker.chunk_audio(audio, transcript)
    
    elif chunk_strategy == "random":
        # Random sampling of chunks
        chunks = []
        audio_length = len(audio)
        
        if audio_length <= chunker.CHUNK_LENGTH_SAMPLES:
            # If audio is short, just return padded version
            return chunker.chunk_audio(audio, transcript)
        
        # Sample 3-5 random chunks
        num_chunks = min(3, max(1, audio_length // chunker.CHUNK_LENGTH_SAMPLES))
        
        for _ in range(num_chunks):
            # Random start position
            max_start = audio_length - chunker.CHUNK_LENGTH_SAMPLES
            start_sample = np.random.randint(0, max_start)
            end_sample = start_sample + chunker.CHUNK_LENGTH_SAMPLES
            
            chunk_audio = audio[start_sample:end_sample]
            start_time = start_sample / chunker.SAMPLE_RATE
            end_time = end_sample / chunker.SAMPLE_RATE
            
            chunk_transcript = chunker._align_transcript(
                transcript, start_time, end_time, 
                audio_length / chunker.SAMPLE_RATE
            )
            
            chunks.append({
                'audio': chunk_audio,
                'start_time': start_time,
                'end_time': end_time,
                'transcript': chunk_transcript,
                'is_padded': False,
                'original_length': chunker.CHUNK_LENGTH_SAMPLES
            })
        
        return chunks
    
    else:
        raise ValueError(f"Unknown chunk strategy: {chunk_strategy}") 