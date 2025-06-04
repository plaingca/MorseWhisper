# Audio Chunking for Whisper Fine-tuning

## Overview

Whisper is designed to process audio in fixed 30-second windows. This poses a challenge when training or evaluating on audio samples longer than 30 seconds. This guide explains our implementation of audio chunking to handle long audio files effectively.

## The Problem

- Whisper expects audio input of exactly 30 seconds (480,000 samples at 16kHz)
- Training data often contains audio samples longer than 30 seconds
- Without proper chunking, long audio would be truncated, losing important information
- Transcripts need to be aligned with audio chunks for accurate training

## Our Solution

We've implemented a comprehensive audio chunking system with the following features:

### 1. Audio Chunking Module (`src/audio_chunking.py`)

The `WhisperAudioChunker` class provides:

- **Automatic chunking** of audio longer than 30 seconds
- **Configurable overlap** between chunks (default: 2 seconds)
- **Automatic padding** for audio shorter than 30 seconds
- **Transcript alignment** to match audio chunks
- **Prediction merging** for evaluation

### 2. Chunking Strategies

Three strategies are available for training:

1. **Sequential** (`--chunk_strategy sequential`)
   - Non-overlapping 30-second chunks
   - Most memory efficient
   - Good for initial training

2. **Sliding** (`--chunk_strategy sliding`)
   - Overlapping chunks with 2-second overlap
   - Better coverage of audio boundaries
   - Recommended for best performance

3. **Random** (`--chunk_strategy random`)
   - Random sampling of chunks from long audio
   - Good for data augmentation
   - Helps model generalize better

### 3. Key Features

#### Padding
- Audio shorter than 30 seconds is zero-padded to exactly 30 seconds
- Padding information is tracked to avoid training on silence

#### Transcript Alignment
- Transcripts are automatically aligned with audio chunks
- Uses simple word-based distribution (works well for morse code's uniform timing)
- Each chunk gets the appropriate portion of the full transcript

#### Overlap Handling
- Overlapping chunks ensure no information is lost at boundaries
- During evaluation, predictions from overlapping chunks are merged intelligently
- Duplicate words at chunk boundaries are automatically removed

## Usage

### Training with Chunking

```bash
python training/fine_tune.py \
    --data_dir data/processed \
    --output_dir models/whisper-morse-chunked \
    --model_name openai/whisper-small \
    --chunk_strategy sliding \
    --batch_size 8 \
    --num_epochs 10
```

### Evaluation with Chunking

```bash
python training/evaluate_model.py \
    --model_path models/whisper-morse-chunked \
    --data_dir data/processed \
    --split test \
    --output_dir evaluation_results
```

To disable chunking during evaluation (for comparison):
```bash
python training/evaluate_model.py \
    --model_path models/whisper-morse-chunked \
    --data_dir data/processed \
    --split test \
    --output_dir evaluation_results \
    --no_chunking
```

## Implementation Details

### Constants
- Sample rate: 16,000 Hz (Whisper requirement)
- Chunk length: 30 seconds (480,000 samples)
- Default overlap: 2 seconds (32,000 samples)
- Minimum chunk size: 1 second (16,000 samples)

### Memory Optimization
- Chunks are created on-the-fly during training to save memory
- Pre-computed chunk metadata reduces processing overhead
- Batch processing of chunks for efficient GPU utilization

### Performance Considerations

1. **Training Impact**
   - Chunked audio creates more training samples
   - Each 60-second audio becomes ~2 chunks with overlap
   - Training time increases proportionally

2. **Evaluation Impact**
   - Chunking adds overhead for long audio
   - Overlapping chunks improve accuracy at boundaries
   - Merging predictions requires additional processing

## Best Practices

1. **For Training**
   - Use `sliding` strategy for best results
   - Consider reducing batch size if memory is limited
   - Monitor chunk distribution in your dataset

2. **For Evaluation**
   - Always use the same chunking strategy as training
   - Compare chunked vs non-chunked performance
   - Analyze WER by audio duration

3. **For Dataset Preparation**
   - Consider splitting very long audio (>2 minutes) at natural boundaries
   - Ensure transcripts are accurate for proper alignment
   - Balance dataset between short and long audio samples

## Debugging

The system provides detailed logging:
- Number of chunks created from each audio file
- Chunk boundaries and overlap information
- Transcript alignment for each chunk
- Padding information

Enable verbose logging by setting the environment variable:
```bash
export WHISPER_CHUNK_DEBUG=1
```

## Performance Metrics

The evaluation script now provides:
- Separate metrics for chunked vs non-chunked audio
- WER analysis by audio duration
- Visual comparison of performance across duration ranges
- Detailed chunking statistics in the evaluation report

## Future Improvements

Potential enhancements to consider:
1. Dynamic overlap based on audio content
2. Smart chunking at silence boundaries
3. Weighted merging based on chunk confidence
4. Multi-scale chunking for very long audio 