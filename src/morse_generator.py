"""
Morse Code Audio Generator

Generates morse code audio signals with realistic CW characteristics:
- Standard morse timing (PARIS = 50 units)
- Variable WPM (words per minute)
- Tone frequency control (typically 600-800 Hz for amateur radio)
- Human-like timing variations
- Envelope shaping for realistic keying
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, List, Tuple, Optional
import soundfile as sf


class MorseCodeGenerator:
    """Generate morse code audio signals with realistic characteristics."""
    
    # Morse code mappings
    MORSE_CODE = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
        'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..', 
        '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
        '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
        '/': '-..-.', '?': '..--..', '.': '.-.-.-', ',': '--..--', '=': '-...-',
        'AR': '.-.-.',  # End of message (often written as +)
        'AS': '.-...',  # Wait
        'BK': '-...-.-',  # Break
        'KN': '-.--.',  # Invitation to transmit to named station only
        'SK': '...-.-',  # End of contact
        'SN': '...-.',  # Understood (often written as VE)
    }
    
    def __init__(self, sample_rate: int = 22050, tone_freq: int = 700):
        """
        Initialize morse code generator.
        
        Args:
            sample_rate: Audio sample rate in Hz
            tone_freq: CW tone frequency in Hz (typically 600-800 Hz)
        """
        self.sample_rate = sample_rate
        self.tone_freq = tone_freq
        
    def text_to_morse(self, text: str) -> str:
        """Convert text to morse code string."""
        morse_text = []
        text = text.upper().strip()
        
        i = 0
        while i < len(text):
            # Check for prosigns (2-character codes)
            if i < len(text) - 1:
                two_char = text[i:i+2]
                if two_char in self.MORSE_CODE:
                    morse_text.append(self.MORSE_CODE[two_char])
                    i += 2
                    continue
            
            # Single character
            char = text[i]
            if char in self.MORSE_CODE:
                morse_text.append(self.MORSE_CODE[char])
            elif char == ' ':
                morse_text.append(' ')  # Word space
            i += 1
            
        return ' '.join(morse_text)
    
    def calculate_timing(self, wpm: int) -> Dict[str, float]:
        """
        Calculate morse timing based on WPM.
        
        Standard timing:
        - Dot = 1 unit
        - Dash = 3 units
        - Inter-element space = 1 unit
        - Inter-character space = 3 units
        - Inter-word space = 7 units
        
        Args:
            wpm: Words per minute (based on PARIS = 50 units)
            
        Returns:
            Dictionary with timing values in seconds
        """
        # PARIS = 50 units, so at X WPM, unit time = 60 / (50 * X)
        unit_time = 60.0 / (50 * wpm)
        
        return {
            'dot': unit_time,
            'dash': unit_time * 3,
            'element_space': unit_time,
            'char_space': unit_time * 3,
            'word_space': unit_time * 7
        }
    
    def apply_envelope(self, signal: np.ndarray, rise_time_ms: float = 5) -> np.ndarray:
        """
        Apply envelope shaping to avoid key clicks.
        
        Args:
            signal: Input signal
            rise_time_ms: Rise/fall time in milliseconds
            
        Returns:
            Shaped signal
        """
        rise_samples = int(rise_time_ms * 0.001 * self.sample_rate)
        if rise_samples > len(signal) // 2:
            rise_samples = len(signal) // 2
            
        if rise_samples > 0:
            # Create raised cosine envelope
            rise = 0.5 * (1 - np.cos(np.pi * np.arange(rise_samples) / rise_samples))
            fall = 0.5 * (1 + np.cos(np.pi * np.arange(rise_samples) / rise_samples))
            
            signal[:rise_samples] *= rise
            signal[-rise_samples:] *= fall
            
        return signal
    
    def generate_tone(self, duration: float, apply_shaping: bool = True) -> np.ndarray:
        """
        Generate a CW tone.
        
        Args:
            duration: Tone duration in seconds
            apply_shaping: Whether to apply envelope shaping
            
        Returns:
            Audio signal
        """
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        tone = np.sin(2 * np.pi * self.tone_freq * t)
        
        if apply_shaping:
            tone = self.apply_envelope(tone)
            
        return tone
    
    def add_timing_variation(self, base_timing: Dict[str, float], 
                           variation_percent: float = 5) -> Dict[str, float]:
        """
        Add human-like timing variations.
        
        Args:
            base_timing: Base timing dictionary
            variation_percent: Maximum variation as percentage
            
        Returns:
            Timing with variations
        """
        varied_timing = {}
        for key, value in base_timing.items():
            # Add random variation
            variation = 1 + (np.random.random() - 0.5) * 2 * (variation_percent / 100)
            varied_timing[key] = value * variation
        return varied_timing
    
    def morse_to_audio(self, morse_code: str, wpm: int = 20, 
                      timing_variation: float = 5) -> np.ndarray:
        """
        Convert morse code string to audio.
        
        Args:
            morse_code: Morse code string (dots, dashes, spaces)
            wpm: Words per minute
            timing_variation: Timing variation percentage (0 for perfect timing)
            
        Returns:
            Audio signal
        """
        base_timing = self.calculate_timing(wpm)
        audio_segments = []
        
        # Split into characters
        characters = morse_code.split(' ')
        
        for i, char in enumerate(characters):
            if not char:  # Word space
                silence_duration = base_timing['word_space']
                if timing_variation > 0:
                    silence_duration *= 1 + (np.random.random() - 0.5) * timing_variation / 50
                audio_segments.append(np.zeros(int(silence_duration * self.sample_rate)))
                continue
            
            # Get timing for this character (with variations)
            if timing_variation > 0:
                timing = self.add_timing_variation(base_timing, timing_variation)
            else:
                timing = base_timing
            
            # Generate elements
            for j, element in enumerate(char):
                if element == '.':
                    audio_segments.append(self.generate_tone(timing['dot']))
                elif element == '-':
                    audio_segments.append(self.generate_tone(timing['dash']))
                
                # Add inter-element space (except after last element)
                if j < len(char) - 1:
                    audio_segments.append(np.zeros(int(timing['element_space'] * self.sample_rate)))
            
            # Add inter-character space (except after last character)
            if i < len(characters) - 1:
                audio_segments.append(np.zeros(int(timing['char_space'] * self.sample_rate)))
        
        return np.concatenate(audio_segments)
    
    def text_to_audio(self, text: str, wpm: int = 20, 
                     timing_variation: float = 5) -> np.ndarray:
        """
        Convert text directly to morse audio.
        
        Args:
            text: Text to convert
            wpm: Words per minute
            timing_variation: Timing variation percentage
            
        Returns:
            Audio signal
        """
        morse_code = self.text_to_morse(text)
        return self.morse_to_audio(morse_code, wpm, timing_variation)
    
    def save_audio(self, audio: np.ndarray, filename: str):
        """Save audio to file."""
        sf.write(filename, audio, self.sample_rate)
        

def generate_sample():
    """Generate a sample morse code audio file."""
    generator = MorseCodeGenerator()
    
    # Sample text
    text = "CQ CQ CQ DE W1ABC W1ABC K"
    
    # Generate audio
    audio = generator.text_to_audio(text, wpm=25, timing_variation=3)
    
    # Save
    generator.save_audio(audio, "sample_cw.wav")
    print(f"Generated morse code for: {text}")
    print(f"Morse: {generator.text_to_morse(text)}")
    

if __name__ == "__main__":
    generate_sample() 