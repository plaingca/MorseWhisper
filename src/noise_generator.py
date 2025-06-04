"""
Noise and Interference Generator for Morse Code Audio

Adds realistic conditions to morse code audio:
- White noise (band noise)
- QRM (interference from other stations)
- QRN (atmospheric noise, static crashes)
- QSB (fading)
- Frequency drift
- Audio filtering effects
- Adjacent channel interference
"""

import numpy as np
import scipy.signal as signal
from typing import Optional, Tuple, List
import random


class NoiseGenerator:
    """Add realistic noise and propagation effects to morse code audio."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize noise generator.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    def add_white_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add white noise to achieve target SNR.
        
        Args:
            audio: Input audio signal
            snr_db: Target signal-to-noise ratio in dB
            
        Returns:
            Audio with added noise
        """
        # Calculate signal power
        signal_power = np.mean(audio ** 2)
        
        # Calculate required noise power
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate white noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        
        return audio + noise
    
    def add_band_limited_noise(self, audio: np.ndarray, snr_db: float,
                              low_freq: float = 300, high_freq: float = 3000) -> np.ndarray:
        """
        Add band-limited noise (more realistic for radio).
        
        Args:
            audio: Input audio signal
            snr_db: Target SNR in dB
            low_freq: Low frequency cutoff in Hz
            high_freq: High frequency cutoff in Hz
            
        Returns:
            Audio with band-limited noise
        """
        # Generate white noise
        noise = np.random.normal(0, 1, len(audio))
        
        # Design bandpass filter
        nyquist = self.sample_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Create bandpass filter
        sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
        filtered_noise = signal.sosfilt(sos, noise)
        
        # Normalize and scale to achieve target SNR
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        filtered_noise = filtered_noise * np.sqrt(noise_power / np.mean(filtered_noise ** 2))
        
        return audio + filtered_noise
    
    def add_qrm(self, audio: np.ndarray, num_interferers: int = 1,
                strength_range: Tuple[float, float] = (0.1, 0.5)) -> np.ndarray:
        """
        Add QRM (interference from other CW stations).
        
        Args:
            audio: Input audio signal
            num_interferers: Number of interfering signals
            strength_range: Range of interference strength (0-1)
            
        Returns:
            Audio with QRM
        """
        result = audio.copy()
        
        for _ in range(num_interferers):
            # Random offset frequency for interferer
            offset_freq = random.uniform(-500, 500)  # Hz offset from main signal
            
            # Generate interference tone
            t = np.arange(len(audio)) / self.sample_rate
            interference = np.sin(2 * np.pi * (700 + offset_freq) * t)
            
            # Random on/off pattern (simulating another CW signal)
            pattern_length = random.randint(100, 500)
            pattern = np.random.choice([0, 1], size=pattern_length, p=[0.6, 0.4])
            
            # Repeat pattern to match audio length
            full_pattern = np.tile(pattern, len(audio) // pattern_length + 1)[:len(audio)]
            
            # Smooth transitions
            smoothed_pattern = signal.savgol_filter(full_pattern.astype(float), 51, 3)
            smoothed_pattern = np.clip(smoothed_pattern, 0, 1)
            
            # Apply pattern to interference
            interference *= smoothed_pattern
            
            # Random strength
            strength = random.uniform(*strength_range)
            result += interference * strength
        
        return result
    
    def add_qrn(self, audio: np.ndarray, crash_rate: float = 0.1,
                crash_strength: float = 2.0) -> np.ndarray:
        """
        Add QRN (atmospheric noise, static crashes).
        
        Args:
            audio: Input audio signal
            crash_rate: Average crashes per second
            crash_strength: Strength of crashes relative to signal
            
        Returns:
            Audio with QRN
        """
        result = audio.copy()
        
        # Calculate number of crashes
        duration = len(audio) / self.sample_rate
        num_crashes = int(np.random.poisson(crash_rate * duration))
        
        for _ in range(num_crashes):
            # Random position for crash
            position = random.randint(0, len(audio) - 1000)
            
            # Generate crash (short burst of noise)
            crash_duration = random.randint(50, 200)  # samples
            crash = np.random.normal(0, crash_strength, crash_duration)
            
            # Apply exponential decay
            decay = np.exp(-np.linspace(0, 5, crash_duration))
            crash *= decay
            
            # Add crash to audio
            end_pos = min(position + crash_duration, len(audio))
            result[position:end_pos] += crash[:end_pos - position]
        
        return result
    
    def add_qsb(self, audio: np.ndarray, fade_rate: float = 0.5,
                fade_depth: float = 0.7, fade_type: str = 'slow') -> np.ndarray:
        """
        Add QSB (signal fading).
        
        Args:
            audio: Input audio signal
            fade_rate: Fading rate in Hz
            fade_depth: Maximum fade depth (0-1, where 1 is complete fade)
            fade_type: 'slow', 'fast', or 'selective'
            
        Returns:
            Audio with fading
        """
        t = np.arange(len(audio)) / self.sample_rate
        
        if fade_type == 'slow':
            # Simple sinusoidal fading
            fade_envelope = 1 - fade_depth * (0.5 + 0.5 * np.sin(2 * np.pi * fade_rate * t))
        elif fade_type == 'fast':
            # Flutter fading (aircraft, aurora)
            fade_envelope = 1 - fade_depth * (0.5 + 0.5 * np.sin(2 * np.pi * fade_rate * 10 * t))
        else:  # selective
            # Multiple fading components (multipath)
            fade_envelope = 1
            for i in range(3):
                rate = fade_rate * (i + 1) * random.uniform(0.8, 1.2)
                depth = fade_depth * random.uniform(0.3, 1.0)
                phase = random.uniform(0, 2 * np.pi)
                fade_envelope *= 1 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t + phase))
        
        return audio * fade_envelope
    
    def add_frequency_drift(self, audio: np.ndarray, max_drift_hz: float = 50,
                           drift_rate_hz_per_sec: float = 5) -> np.ndarray:
        """
        Add frequency drift (VFO drift, Doppler).
        
        Args:
            audio: Input audio signal
            max_drift_hz: Maximum drift in Hz
            drift_rate_hz_per_sec: Drift rate
            
        Returns:
            Audio with frequency drift
        """
        # Generate drift pattern
        t = np.arange(len(audio)) / self.sample_rate
        
        # Random walk drift
        num_points = int(len(audio) / self.sample_rate) + 1
        drift_points = np.cumsum(np.random.randn(num_points) * drift_rate_hz_per_sec)
        drift_points = np.clip(drift_points, -max_drift_hz, max_drift_hz)
        
        # Interpolate to match audio length
        drift_curve = np.interp(t, np.linspace(0, t[-1], num_points), drift_points)
        
        # Apply frequency shift using phase modulation
        phase = 2 * np.pi * np.cumsum(drift_curve) / self.sample_rate
        
        # Hilbert transform for single sideband modulation
        analytic_signal = signal.hilbert(audio)
        shifted_signal = np.real(analytic_signal * np.exp(1j * phase))
        
        return shifted_signal
    
    def add_filter_effects(self, audio: np.ndarray, 
                          filter_type: str = 'cw_narrow') -> np.ndarray:
        """
        Apply typical receiver filtering.
        
        Args:
            audio: Input audio signal
            filter_type: 'cw_narrow', 'cw_wide', 'ssb'
            
        Returns:
            Filtered audio
        """
        nyquist = self.sample_rate / 2
        
        if filter_type == 'cw_narrow':
            # Narrow CW filter (250 Hz bandwidth)
            low_freq = 575
            high_freq = 825
        elif filter_type == 'cw_wide':
            # Wide CW filter (500 Hz bandwidth)
            low_freq = 450
            high_freq = 950
        else:  # ssb
            # SSB filter
            low_freq = 300
            high_freq = 2700
        
        # Design bandpass filter
        sos = signal.butter(6, [low_freq / nyquist, high_freq / nyquist], 
                           btype='band', output='sos')
        
        return signal.sosfilt(sos, audio)
    
    def add_agc_effects(self, audio: np.ndarray, attack_time: float = 0.01,
                       release_time: float = 0.5) -> np.ndarray:
        """
        Simulate AGC (Automatic Gain Control) effects.
        
        Args:
            audio: Input audio signal
            attack_time: AGC attack time in seconds
            release_time: AGC release time in seconds
            
        Returns:
            Audio with AGC effects
        """
        # Simple envelope follower AGC
        envelope = np.abs(audio)
        
        # Smooth envelope
        attack_samples = int(attack_time * self.sample_rate)
        release_samples = int(release_time * self.sample_rate)
        
        agc_envelope = np.zeros_like(envelope)
        agc_envelope[0] = envelope[0]
        
        for i in range(1, len(envelope)):
            if envelope[i] > agc_envelope[i-1]:
                # Attack
                alpha = 1 - np.exp(-1 / attack_samples)
                agc_envelope[i] = alpha * envelope[i] + (1 - alpha) * agc_envelope[i-1]
            else:
                # Release
                alpha = 1 - np.exp(-1 / release_samples)
                agc_envelope[i] = alpha * envelope[i] + (1 - alpha) * agc_envelope[i-1]
        
        # Avoid division by zero
        agc_envelope = np.maximum(agc_envelope, 1e-6)
        
        # Apply compression
        target_level = np.mean(agc_envelope)
        gain = target_level / agc_envelope
        gain = np.clip(gain, 0.1, 10)  # Limit gain range
        
        return audio * gain
    
    def add_multipath(self, audio: np.ndarray, num_paths: int = 2,
                     max_delay_ms: float = 5, max_attenuation: float = 0.5) -> np.ndarray:
        """
        Add multipath propagation effects.
        
        Args:
            audio: Input audio signal
            num_paths: Number of propagation paths
            max_delay_ms: Maximum delay in milliseconds
            max_attenuation: Maximum attenuation for delayed paths
            
        Returns:
            Audio with multipath effects
        """
        result = audio.copy()
        
        for i in range(1, num_paths):
            # Random delay and attenuation for this path
            delay_ms = random.uniform(0.1, max_delay_ms)
            delay_samples = int(delay_ms * 0.001 * self.sample_rate)
            attenuation = random.uniform(0.1, max_attenuation)
            
            # Create delayed version
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * attenuation
            
            # Random phase shift
            phase_shift = random.uniform(0, 2 * np.pi)
            delayed *= np.exp(1j * phase_shift).real
            
            result += delayed
        
        return result
    
    def apply_realistic_conditions(self, audio: np.ndarray, 
                                 condition_preset: str = 'contest_good') -> np.ndarray:
        """
        Apply a preset combination of realistic conditions.
        
        Args:
            audio: Input audio signal
            condition_preset: Preset name
            
        Returns:
            Audio with conditions applied
        """
        if condition_preset == 'contest_good':
            # Good contest conditions
            audio = self.add_filter_effects(audio, 'cw_narrow')
            audio = self.add_band_limited_noise(audio, snr_db=20)
            audio = self.add_qrm(audio, num_interferers=1, strength_range=(0.05, 0.15))
            audio = self.add_qsb(audio, fade_rate=0.1, fade_depth=0.2, fade_type='slow')
            
        elif condition_preset == 'contest_moderate':
            # Moderate contest conditions
            audio = self.add_filter_effects(audio, 'cw_narrow')
            audio = self.add_band_limited_noise(audio, snr_db=15)
            audio = self.add_qrm(audio, num_interferers=2, strength_range=(0.1, 0.3))
            audio = self.add_qrn(audio, crash_rate=0.2, crash_strength=1.5)
            audio = self.add_qsb(audio, fade_rate=0.3, fade_depth=0.4, fade_type='slow')
            audio = self.add_frequency_drift(audio, max_drift_hz=30)
            
        elif condition_preset == 'contest_poor':
            # Poor contest conditions
            audio = self.add_filter_effects(audio, 'cw_wide')
            audio = self.add_band_limited_noise(audio, snr_db=10)
            audio = self.add_qrm(audio, num_interferers=3, strength_range=(0.2, 0.5))
            audio = self.add_qrn(audio, crash_rate=0.5, crash_strength=2.0)
            audio = self.add_qsb(audio, fade_rate=0.5, fade_depth=0.6, fade_type='selective')
            audio = self.add_frequency_drift(audio, max_drift_hz=50)
            audio = self.add_multipath(audio, num_paths=3)
            
        elif condition_preset == 'dx_expedition':
            # DX expedition (weak signal, heavy QRM)
            audio = self.add_filter_effects(audio, 'cw_narrow')
            audio = self.add_band_limited_noise(audio, snr_db=8)
            audio = self.add_qrm(audio, num_interferers=5, strength_range=(0.3, 0.7))
            audio = self.add_qsb(audio, fade_rate=0.2, fade_depth=0.7, fade_type='slow')
            audio = self.add_frequency_drift(audio, max_drift_hz=100)
            
        else:  # clean
            # Clean conditions (for testing)
            audio = self.add_filter_effects(audio, 'cw_narrow')
            audio = self.add_band_limited_noise(audio, snr_db=30)
        
        # Always apply AGC at the end
        audio = self.add_agc_effects(audio)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * 0.95 / max_val
            
        return audio


def test_noise_generator():
    """Test the noise generator with a simple tone."""
    import matplotlib.pyplot as plt
    
    # Generate test tone
    sample_rate = 22050
    duration = 2.0
    t = np.arange(int(duration * sample_rate)) / sample_rate
    test_tone = np.sin(2 * np.pi * 700 * t)
    
    # Apply various effects
    generator = NoiseGenerator(sample_rate)
    
    conditions = ['clean', 'contest_good', 'contest_moderate', 'contest_poor']
    
    fig, axes = plt.subplots(len(conditions), 1, figsize=(12, 8))
    
    for i, condition in enumerate(conditions):
        processed = generator.apply_realistic_conditions(test_tone.copy(), condition)
        axes[i].plot(t[:1000], processed[:1000])
        axes[i].set_title(f'Condition: {condition}')
        axes[i].set_ylabel('Amplitude')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('noise_effects_demo.png')
    print("Saved noise effects demo to noise_effects_demo.png")


if __name__ == "__main__":
    test_noise_generator() 