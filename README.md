# MorseWhisper - Fine-tuning Whisper for Amateur Radio Morse Code

This project fine-tunes OpenAI's Whisper model to decode morse code (CW) transmissions typical in amateur radio contests.

## Features

- Synthetic morse code dataset generation with realistic conditions:
  - Amateur radio callsigns (US and international)
  - Contest exchanges (RST reports, grid squares, serial numbers)
  - Realistic noise, QRM (interference), QSB (fading)
  - Timing variations (human-like keying imperfections)
  - Standard CW tone frequencies (600-800 Hz)
- Fine-tuning pipeline for Whisper
- WER (Word Error Rate) evaluation
- Support for various contest formats

## Project Structure

```
MorseWhisper/
├── src/
│   ├── morse_generator.py    # Morse code audio synthesis
│   ├── dataset_builder.py    # Dataset creation with realistic conditions
│   ├── callsign_generator.py # Amateur radio callsign generation
│   ├── contest_exchanges.py  # Contest exchange formats
│   └── noise_generator.py    # Noise and interference simulation
├── training/
│   ├── fine_tune.py         # Whisper fine-tuning script
│   └── evaluate_model.py    # WER evaluation
├── data/
│   ├── raw/                 # Generated audio files
│   ├── processed/           # Processed dataset
│   └── splits/              # Train/val/test splits
├── models/                  # Saved fine-tuned models
├── notebooks/              # Jupyter notebooks for analysis
└── configs/               # Configuration files
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Generate synthetic dataset:
```bash
# Generate a dataset with 10,000 samples (includes 5% noise-only samples)
python src/dataset_builder.py \
    --num_samples 10000 \
    --output_dir data/morse_dataset \
    --noise_ratio 0.05
```

**Performance Note**: Dataset generation now uses multiprocessing for 3-6x faster generation:
- Default: Uses all available CPU cores (up to 8)
- Specify workers: `--num_workers 4`
- Generation rate: ~15-20 samples/second on modern hardware

The generated dataset includes:
- Audio files under 30 seconds (Whisper's context window)
- Varied duration distribution (3-28 seconds)
- 5-10% noise-only samples for robustness
- Realistic contest scenarios and QRM/QRN conditions
- Proper train/validation/test splits

2. Fine-tune Whisper:
```bash
python training/fine_tune.py --model whisper-small --dataset data/processed
```

3. Evaluate:
```bash
python training/evaluate_model.py --model models/best_model --test_set data/splits/test
```

## CW Contest Formats Supported

- CQWW (CQ World Wide)
- ARRL DX
- CQ WPX
- Field Day
- Sweepstakes 