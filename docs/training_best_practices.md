# Whisper Training Best Practices for Morse Code

## Overview

This guide explains the best practices for training Whisper on morse code recognition, focusing on proper audio preparation and when to use different training approaches.

## Key Principle: Natural 30-Second Context Window

Whisper is designed to process audio in 30-second chunks. For optimal training:

1. **Generate training samples under 30 seconds** - This is the cleanest approach
2. **Include noise-only samples** - Helps the model learn to handle silence/noise
3. **Use chunking only when necessary** - For inference on long audio or legacy datasets

## Dataset Generation Best Practices

### 1. Generate Naturally-Sized Samples

```bash
# Generate dataset with samples under 30 seconds
python src/dataset_builder.py \
    --num_samples 10000 \
    --output_dir data/morse_dataset \
    --max_duration 28.0 \
    --noise_ratio 0.05
```

This creates:
- Audio samples between 3-28 seconds
- 5% noise-only samples for robustness
- Proper duration distribution for varied training

### 2. Duration Distribution

The recommended duration distribution:
- **3-8 seconds**: 25% (short transmissions)
- **8-15 seconds**: 35% (medium messages)
- **15-22 seconds**: 30% (longer exchanges)
- **22-28 seconds**: 10% (complete QSOs)

### 3. Include Noise-Only Samples

Always include 5-10% noise-only samples:
- Helps model learn to output nothing when there's no signal
- Improves robustness to noise
- Reduces false positives

## Training Approaches

### Approach 1: Clean Dataset (Recommended)

For datasets where all samples are under 30 seconds:

```bash
python training/fine_tune.py \
    --data_dir data/morse_dataset \
    --output_dir models/whisper-morse-clean \
    --model_name openai/whisper-small \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 1e-5
```

Benefits:
- No chunking complexity
- Natural alignment of audio and transcripts
- Faster training
- Better performance

### Approach 2: Legacy Dataset with Chunking

For existing datasets with long audio files:

```bash
python training/fine_tune.py \
    --data_dir data/legacy_dataset \
    --output_dir models/whisper-morse-chunked \
    --model_name openai/whisper-small \
    --use_chunking \
    --chunk_strategy sliding \
    --batch_size 8 \
    --num_epochs 10
```

Use this when:
- You have existing long audio files
- Regenerating the dataset is not feasible
- You need to maintain compatibility

## Model Configuration

### Special Tokens

The system uses two special tokens:
- `<|morse|>`: Indicates morse code content
- `<|noise|>`: Indicates noise-only audio

These help the model distinguish between signal types.

### Training Parameters

Recommended parameters for morse code:

```python
# For clean datasets (samples < 30s)
--batch_size 16          # Can use larger batches
--learning_rate 1e-5     # Standard learning rate
--warmup_steps 500       # Gradual warmup
--eval_steps 500         # Regular evaluation

# For chunked datasets
--batch_size 8           # Smaller due to more samples
--learning_rate 5e-6     # Lower LR for stability
--warmup_steps 1000      # More warmup needed
```

## Evaluation Best Practices

### 1. Evaluate on Natural Test Set

Always evaluate on a test set with natural audio lengths:

```bash
python training/evaluate_model.py \
    --model_path models/whisper-morse-clean \
    --data_dir data/morse_dataset \
    --split test \
    --output_dir evaluation_results
```

### 2. Test Long Audio Inference

For real-world usage with long audio:

```bash
python training/evaluate_model.py \
    --model_path models/whisper-morse-clean \
    --data_dir data/long_audio_test \
    --split test \
    --output_dir evaluation_long \
    --no_chunking  # Test without chunking first
```

Then with chunking:

```bash
python training/evaluate_model.py \
    --model_path models/whisper-morse-clean \
    --data_dir data/long_audio_test \
    --split test \
    --output_dir evaluation_long_chunked
    # Chunking enabled by default for inference
```

### 3. Key Metrics

Monitor these metrics:
- **WER on morse samples**: Primary accuracy metric
- **Noise accuracy**: How well it handles noise-only samples
- **Duration analysis**: Performance across different audio lengths
- **Real-time factor**: Processing speed

## Common Issues and Solutions

### Issue 1: Poor Performance on Short Samples

**Solution**: Ensure your dataset has enough short samples (3-8 seconds). The model needs to learn when to stop generating text.

### Issue 2: Hallucinations on Noise

**Solution**: Increase the ratio of noise-only samples to 10%. Make sure the model learns to output nothing when there's no signal.

### Issue 3: Errors at Audio Boundaries

**Solution**: 
- For training: Use naturally-sized samples
- For inference: Use sliding window chunking with 2-second overlap

### Issue 4: Slow Training

**Solution**: 
- Use naturally-sized samples (no chunking)
- Enable FP16 training with `--fp16`
- Increase batch size if memory allows

## Production Deployment

### 1. Model Selection

Choose based on your needs:
- **whisper-tiny**: Fastest, good for real-time
- **whisper-small**: Best balance of speed/accuracy
- **whisper-medium**: Higher accuracy, slower

### 2. Inference Pipeline

For production inference on arbitrary-length audio:

```python
from src.audio_chunking import WhisperAudioChunker
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load model
processor = WhisperProcessor.from_pretrained("models/whisper-morse-clean")
model = WhisperForConditionalGeneration.from_pretrained("models/whisper-morse-clean")

# For long audio, use chunking
chunker = WhisperAudioChunker(overlap_seconds=2.0)

# Process audio
if audio_duration > 30:
    chunks = chunker.chunk_audio(audio)
    # Process each chunk and merge results
else:
    # Process directly
    result = model.generate(audio)
```

### 3. Optimization

- Use ONNX export for faster inference
- Implement streaming for real-time processing
- Cache results for repeated audio segments

## Summary

1. **Prefer naturally-sized training samples** (<30 seconds)
2. **Include noise-only samples** (5-10% of dataset)
3. **Use chunking only for inference** on long audio
4. **Monitor performance across duration ranges**
5. **Test both chunked and non-chunked inference**

This approach provides the best balance of training simplicity, model performance, and real-world applicability. 