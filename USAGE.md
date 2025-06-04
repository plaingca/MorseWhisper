# MorseWhisper Usage Guide

This guide explains how to use MorseWhisper to fine-tune Whisper for amateur radio morse code recognition.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Components

First, verify all components are working:

```bash
python test_components.py
```

This will:
- Test morse code generation
- Test callsign generation
- Test contest exchange generation
- Test noise/interference simulation
- Generate sample audio files

### 3. Run Complete Pipeline

To run the entire pipeline with default settings:

```bash
python run_pipeline.py --num_samples 10000
```

This will:
1. Generate 10,000 synthetic morse code samples
2. Fine-tune Whisper on the dataset
3. Evaluate the model and compare with baseline

## Detailed Usage

### Dataset Generation Only

Generate a custom dataset without training:

```bash
python src/dataset_builder.py \
    --num_samples 50000 \
    --output_dir data/my_dataset \
    --config configs/contest_cw.yaml
```

Parameters:
- `--num_samples`: Number of audio samples to generate
- `--output_dir`: Where to save the dataset
- `--config`: Configuration file with generation parameters
- `--sample_rate`: Audio sample rate (default: 16000 for Whisper)

### Fine-tuning Only

Fine-tune Whisper on an existing dataset:

```bash
python training/fine_tune.py \
    --model_name openai/whisper-small \
    --data_dir data/my_dataset \
    --output_dir models/my_model \
    --batch_size 16 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --fp16
```

Available Whisper models:
- `openai/whisper-tiny` (39M parameters)
- `openai/whisper-base` (74M parameters)
- `openai/whisper-small` (244M parameters) - Recommended
- `openai/whisper-medium` (769M parameters)
- `openai/whisper-large` (1550M parameters)

### Evaluation Only

Evaluate a trained model:

```bash
python training/evaluate_model.py \
    --model_path models/my_model \
    --data_dir data/my_dataset \
    --output_dir evaluation_results \
    --compare_baseline
```

This generates:
- WER/CER metrics
- Performance plots
- Detailed error analysis
- Comparison with baseline Whisper

### Inference

Use the fine-tuned model to decode morse code:

```bash
# Single file
python inference.py \
    --model_path models/my_model \
    --audio_file path/to/morse.wav

# Directory of files
python inference.py \
    --model_path models/my_model \
    --audio_dir path/to/audio_files \
    --output results.json \
    --output_format json
```

## Configuration

The main configuration file `configs/contest_cw.yaml` controls:

### Contest Types
- CQWW (CQ World Wide)
- CQWPX (CQ WPX)
- ARRLDX (ARRL DX)
- FIELD_DAY
- NAQP
- SWEEPSTAKES
- SPRINT
- IARU

### Signal Conditions
- `clean`: Perfect conditions (baseline)
- `contest_good`: SNR ~20dB, light QRM
- `contest_moderate`: SNR ~15dB, moderate QRM/QRN
- `contest_poor`: SNR ~10dB, heavy interference
- `dx_expedition`: Weak signal, pileup conditions

### Speed Range
- Default: 15-40 WPM
- Configurable in `wpm_range`

### Audio Parameters
- Tone frequency: 600-800 Hz (typical CW range)
- Timing variations: 2-8% (human-like)
- Sample rate: 16kHz (Whisper standard)

## Advanced Usage

### Custom Pipeline

Create a custom pipeline with specific settings:

```bash
python run_pipeline.py \
    --config configs/my_custom_config.yaml \
    --num_samples 25000 \
    --model_name openai/whisper-base \
    --batch_size 32 \
    --num_epochs 5 \
    --output_base experiments/my_experiment
```

### Resume Training

Resume from a checkpoint:

```bash
python training/fine_tune.py \
    --model_name openai/whisper-small \
    --data_dir data/my_dataset \
    --output_dir models/my_model \
    --resume_from_checkpoint models/my_model/checkpoint-1500
```

### Multi-GPU Training

For multi-GPU training, use Hugging Face's accelerate:

```bash
accelerate config  # Configure multi-GPU settings
accelerate launch training/fine_tune.py --arguments...
```

## Performance Tips

### For Best WER Performance

1. **Dataset Size**: Use at least 10,000 samples, preferably 50,000+
2. **Model Size**: whisper-small or whisper-medium work best
3. **Training**: 
   - Use mixed precision (--fp16)
   - Learning rate: 1e-5 to 5e-5
   - Batch size: As large as GPU memory allows
4. **Data Variety**: Include all contest types and conditions

### For Fast Training

1. Use smaller model (whisper-tiny or whisper-base)
2. Reduce dataset size for prototyping
3. Use fp16 training
4. Increase batch size if GPU memory allows

### For Inference Speed

1. Use smaller models for real-time applications
2. Consider quantization (not implemented yet)
3. Batch process multiple files

## Typical Results

With proper training, expect:
- **Clean conditions**: <5% WER
- **Good contest conditions**: 5-10% WER
- **Moderate conditions**: 10-20% WER
- **Poor conditions**: 20-40% WER

The model performs best on:
- Standard callsigns
- Common contest exchanges
- Speeds between 20-30 WPM

Challenging scenarios:
- Very high speed (>35 WPM)
- Heavy QRM/QRN
- Unusual callsigns or abbreviations
- Weak signals with QSB

## Troubleshooting

### Out of Memory

Reduce batch size or use a smaller model:
```