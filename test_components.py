"""
Test MorseWhisper Components

Quick tests to verify all components are working correctly.
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from morse_generator import MorseCodeGenerator
from callsign_generator import CallsignGenerator
from contest_exchanges import ContestExchangeGenerator
from noise_generator import NoiseGenerator


def test_morse_generator():
    """Test morse code generation."""
    print("Testing Morse Code Generator...")
    
    generator = MorseCodeGenerator(sample_rate=16000)
    
    # Test text to morse conversion
    test_texts = [
        "CQ CQ CQ DE W1ABC K",
        "599 14",
        "TU 73 GL",
        "W1ABC DE K3XYZ 599 05 K"
    ]
    
    for text in test_texts:
        morse = generator.text_to_morse(text)
        print(f"Text: {text}")
        print(f"Morse: {morse}\n")
        
        # Generate audio
        audio = generator.text_to_audio(text, wpm=25, timing_variation=5)
        print(f"Generated {len(audio)/16000:.2f} seconds of audio\n")
    
    # Save a sample
    sample_audio = generator.text_to_audio("TEST DE W1ABC W1ABC K", wpm=20)
    generator.save_audio(sample_audio, "test_morse.wav")
    print("✅ Morse generator test passed! Sample saved to test_morse.wav\n")


def test_callsign_generator():
    """Test callsign generation."""
    print("Testing Callsign Generator...")
    
    generator = CallsignGenerator()
    
    # Generate various callsigns
    print("Random callsigns:")
    for _ in range(10):
        call, country = generator.generate_callsign()
        print(f"  {call:10s} ({country})")
    
    print("\nContest stations:")
    for _ in range(5):
        call, _ = generator.generate_callsign(special_format='CONTEST_STATION')
        print(f"  {call}")
    
    print("\nPortable stations:")
    base = "W1ABC"
    print(f"  {generator.generate_portable_callsign(base, 'FL')}")
    print(f"  {generator.generate_portable_callsign(base)}")
    
    print("✅ Callsign generator test passed!\n")


def test_exchange_generator():
    """Test contest exchange generation."""
    print("Testing Contest Exchange Generator...")
    
    generator = ContestExchangeGenerator()
    
    contests = ['CQWW', 'CQWPX', 'ARRLDX', 'FIELD_DAY', 'NAQP']
    
    for contest in contests:
        print(f"\n{contest}:")
        for _ in range(3):
            exchange = generator.generate_exchange(contest)
            formatted = generator.format_exchange_string(exchange, contest)
            print(f"  {formatted}")
    
    print("\nGrid squares:")
    for _ in range(5):
        print(f"  {generator.generate_grid_square()}")
    
    print("✅ Exchange generator test passed!\n")


def test_noise_generator():
    """Test noise and interference generation."""
    print("Testing Noise Generator...")
    
    # Generate test tone
    sample_rate = 16000
    duration = 3.0
    t = np.arange(int(duration * sample_rate)) / sample_rate
    clean_tone = np.sin(2 * np.pi * 700 * t)
    
    generator = NoiseGenerator(sample_rate)
    
    # Test different conditions
    conditions = ['clean', 'contest_good', 'contest_moderate', 'contest_poor', 'dx_expedition']
    
    plt.figure(figsize=(12, 10))
    
    for i, condition in enumerate(conditions):
        processed = generator.apply_realistic_conditions(clean_tone.copy(), condition)
        
        # Plot time domain
        plt.subplot(len(conditions), 2, i*2+1)
        plt.plot(t[:500], processed[:500])
        plt.title(f'{condition} - Time Domain')
        plt.ylabel('Amplitude')
        if i == len(conditions)-1:
            plt.xlabel('Time (s)')
        
        # Plot frequency domain
        plt.subplot(len(conditions), 2, i*2+2)
        fft = np.fft.fft(processed[:8192])
        freqs = np.fft.fftfreq(8192, 1/sample_rate)
        plt.plot(freqs[:4096], 20*np.log10(np.abs(fft[:4096])))
        plt.title(f'{condition} - Frequency Domain')
        plt.ylabel('Magnitude (dB)')
        plt.xlim(0, 2000)
        plt.ylim(-60, 40)
        if i == len(conditions)-1:
            plt.xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig('test_noise_conditions.png', dpi=150)
    print("✅ Noise generator test passed! Plots saved to test_noise_conditions.png\n")


def test_full_generation():
    """Test full morse code generation with realistic conditions."""
    print("Testing Full Morse Code Generation Pipeline...")
    
    # Initialize all components
    morse_gen = MorseCodeGenerator(sample_rate=16000)
    callsign_gen = CallsignGenerator()
    exchange_gen = ContestExchangeGenerator()
    noise_gen = NoiseGenerator(sample_rate=16000)
    
    # Generate a complete QSO
    my_call, _ = callsign_gen.generate_callsign(country='USA')
    dx_call, _ = callsign_gen.generate_callsign(country='GERMANY')
    
    # Create contest exchange
    contest = 'CQWW'
    exchange = exchange_gen.generate_exchange(contest)
    exchange_str = exchange_gen.format_exchange_string(exchange, contest)
    
    # Build QSO text
    qso_text = f"CQ TEST DE {my_call} {my_call} K"
    print(f"QSO Text: {qso_text}")
    
    # Generate morse audio
    clean_audio = morse_gen.text_to_audio(qso_text, wpm=25, timing_variation=5)
    
    # Apply realistic conditions
    conditions = ['contest_good', 'contest_moderate', 'contest_poor']
    
    for i, condition in enumerate(conditions):
        processed_audio = noise_gen.apply_realistic_conditions(clean_audio.copy(), condition)
        
        # Save audio
        filename = f"test_qso_{condition}.wav"
        morse_gen.save_audio(processed_audio, filename)
        print(f"  Saved: {filename}")
    
    print("✅ Full generation test passed!\n")


def main():
    """Run all component tests."""
    print("🧪 Testing MorseWhisper Components\n")
    print("="*60)
    
    try:
        test_morse_generator()
        test_callsign_generator()
        test_exchange_generator()
        test_noise_generator()
        test_full_generation()
        
        print("="*60)
        print("✅ All tests passed! Ready to run full pipeline.")
        print("\nTo generate a dataset and train, run:")
        print("  python run_pipeline.py --num_samples 1000")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 