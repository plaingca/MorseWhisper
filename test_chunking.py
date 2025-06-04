"""
Test script for audio chunking implementation

This script validates that the chunking implementation works correctly
for various audio lengths and configurations.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from audio_chunking import WhisperAudioChunker, create_training_chunks


def test_short_audio_padding():
    """Test that short audio is properly padded to 30 seconds."""
    print("\n=== Testing short audio padding ===")
    
    chunker = WhisperAudioChunker()
    
    # Create 10-second audio
    audio = np.random.randn(10 * 16000)
    transcript = "CQ TEST DE K1ABC K1ABC K"
    
    chunks = chunker.chunk_audio(audio, transcript)
    
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert len(chunks[0]['audio']) == 480000, f"Expected 480000 samples, got {len(chunks[0]['audio'])}"
    assert chunks[0]['is_padded'] == True, "Chunk should be marked as padded"
    assert chunks[0]['transcript'] == transcript, "Transcript should be preserved"
    
    print(f"✓ Short audio (10s) correctly padded to 30s")
    print(f"  Original length: {len(audio)} samples")
    print(f"  Padded length: {len(chunks[0]['audio'])} samples")


def test_exact_30s_audio():
    """Test that 30-second audio is processed without modification."""
    print("\n=== Testing exact 30s audio ===")
    
    chunker = WhisperAudioChunker()
    
    # Create exactly 30-second audio
    audio = np.random.randn(30 * 16000)
    transcript = "TEST MESSAGE EXACTLY THIRTY SECONDS"
    
    chunks = chunker.chunk_audio(audio, transcript)
    
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert len(chunks[0]['audio']) == 480000, f"Expected 480000 samples, got {len(chunks[0]['audio'])}"
    assert chunks[0]['is_padded'] == True, "30s audio is still considered padded in a single chunk"
    
    print(f"✓ 30s audio processed correctly")


def test_long_audio_chunking():
    """Test that long audio is properly chunked with overlap."""
    print("\n=== Testing long audio chunking ===")
    
    chunker = WhisperAudioChunker(overlap_seconds=2.0)
    
    # Create 75-second audio
    audio = np.random.randn(75 * 16000)
    transcript = " ".join([f"WORD{i}" for i in range(50)])
    
    chunks = chunker.chunk_audio(audio, transcript)
    
    print(f"✓ 75s audio chunked into {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s")
        print(f"    Transcript: {chunk['transcript'][:50]}...")
        print(f"    Padded: {chunk['is_padded']}")
    
    # Verify chunk properties
    assert len(chunks) == 3, f"Expected 3 chunks for 75s audio, got {len(chunks)}"
    
    # Check overlap
    for i in range(len(chunks) - 1):
        overlap = chunks[i]['end_time'] - chunks[i+1]['start_time']
        assert abs(overlap - 2.0) < 0.1, f"Expected 2s overlap, got {overlap:.2f}s"


def test_transcript_alignment():
    """Test that transcripts are properly aligned with chunks."""
    print("\n=== Testing transcript alignment ===")
    
    chunker = WhisperAudioChunker()
    
    # Create 60-second audio with known transcript
    audio = np.random.randn(60 * 16000)
    words = [f"WORD{i:02d}" for i in range(40)]
    transcript = " ".join(words)
    
    chunks = chunker.chunk_audio(audio, transcript)
    
    print(f"✓ Transcript alignment for {len(chunks)} chunks:")
    
    all_chunk_words = []
    for i, chunk in enumerate(chunks):
        chunk_words = chunk['transcript'].split()
        all_chunk_words.extend(chunk_words)
        print(f"  Chunk {i}: {len(chunk_words)} words")
        print(f"    First word: {chunk_words[0] if chunk_words else 'None'}")
        print(f"    Last word: {chunk_words[-1] if chunk_words else 'None'}")
    
    # Verify all words are covered (with possible duplicates due to overlap)
    unique_words = set(all_chunk_words)
    original_words = set(words)
    assert original_words.issubset(unique_words), "Some words missing from chunks"


def test_chunk_merging():
    """Test that predictions from chunks can be properly merged."""
    print("\n=== Testing chunk prediction merging ===")
    
    chunker = WhisperAudioChunker()
    
    # Simulate predictions from overlapping chunks
    predictions = [
        {
            'text': "CQ TEST DE K1ABC K1ABC",
            'start_time': 0.0,
            'end_time': 30.0
        },
        {
            'text': "K1ABC K1ABC K DE W2XYZ",
            'start_time': 28.0,
            'end_time': 58.0
        },
        {
            'text': "W2XYZ 599 001 001",
            'start_time': 56.0,
            'end_time': 86.0
        }
    ]
    
    merged = chunker.merge_chunk_predictions(predictions)
    print(f"✓ Merged prediction: {merged}")
    
    # Check that duplicates at boundaries are removed
    assert "K1ABC K1ABC K1ABC K1ABC" not in merged, "Duplicates should be removed"
    assert "CQ TEST DE K1ABC" in merged, "Beginning should be preserved"
    assert "599 001 001" in merged, "End should be preserved"


def test_different_chunk_strategies():
    """Test different chunking strategies."""
    print("\n=== Testing chunking strategies ===")
    
    # Create 90-second audio
    audio = np.random.randn(90 * 16000)
    transcript = " ".join([f"MSG{i}" for i in range(60)])
    
    strategies = ["sequential", "sliding", "random"]
    
    for strategy in strategies:
        chunks = create_training_chunks(audio, transcript, chunk_strategy=strategy)
        print(f"\n✓ {strategy.capitalize()} strategy: {len(chunks)} chunks")
        
        if strategy == "sequential":
            # Should have no overlap
            for i in range(len(chunks) - 1):
                assert chunks[i]['end_time'] <= chunks[i+1]['start_time'], \
                    "Sequential chunks should not overlap"
        
        elif strategy == "sliding":
            # Should have overlap
            for i in range(len(chunks) - 1):
                overlap = chunks[i]['end_time'] - chunks[i+1]['start_time']
                assert overlap > 0, "Sliding chunks should overlap"
        
        elif strategy == "random":
            # Should have random positions
            start_times = [c['start_time'] for c in chunks]
            assert len(set(start_times)) == len(start_times), \
                "Random chunks should have different start times"


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== Testing edge cases ===")
    
    chunker = WhisperAudioChunker()
    
    # Test empty audio
    empty_audio = np.array([])
    chunks = chunker.chunk_audio(empty_audio, "")
    assert len(chunks) == 1, "Empty audio should produce one padded chunk"
    assert len(chunks[0]['audio']) == 480000, "Empty audio should be padded to 30s"
    print("✓ Empty audio handled correctly")
    
    # Test very short audio (< 1 second)
    short_audio = np.random.randn(8000)  # 0.5 seconds
    chunks = chunker.chunk_audio(short_audio, "TEST")
    assert len(chunks) == 1, "Very short audio should produce one chunk"
    assert chunks[0]['is_padded'] == True, "Very short audio should be padded"
    print("✓ Very short audio handled correctly")
    
    # Test audio just over 30 seconds
    audio_31s = np.random.randn(31 * 16000)
    chunks = chunker.chunk_audio(audio_31s, "JUST OVER THIRTY SECONDS")
    assert len(chunks) == 2, "31s audio should produce 2 chunks with overlap"
    print("✓ Audio just over 30s handled correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Audio Chunking Tests")
    print("=" * 60)
    
    try:
        test_short_audio_padding()
        test_exact_30s_audio()
        test_long_audio_chunking()
        test_transcript_alignment()
        test_chunk_merging()
        test_different_chunk_strategies()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests() 