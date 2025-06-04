# .gitignore Notes

## Overview
The `.gitignore` file for MorseWhisper is structured to:
- Keep source code and documentation
- Ignore generated models, datasets, and outputs
- Prevent large binary files from being committed

## Key Decisions

### What We Keep
- **Source code**: All `.py` files in `src/`, `training/`, `scripts/`
- **Documentation**: `README.md`, `USAGE.md`, and all markdown in `docs/`
- **Configuration**: `requirements*.txt`, config files
- **Test audio**: Root-level test WAV files for demos

### What We Ignore

#### Models & Training Outputs
- Model checkpoints and weights (`.bin`, `.safetensors`)
- Training runs and logs
- Experiment outputs in `experiments/*/model*/`

#### Datasets
- Audio files in dataset directories
- Generated morse datasets (`morse_dataset_*/`)
- Large CSV files with audio data

#### Temporary Files
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environments (`.env`, `venv/`)
- IDE settings (`.vscode/`, `.idea/`)
- Temporary summaries (`*_SUMMARY.md`)

#### Large Files
- Audio files (`.wav`, `.mp3`, etc.) except test files
- Archives (`.zip`, `.tar.gz`)
- Model weights and checkpoints

## Special Cases

### Exceptions
- `!test_*.wav` - Keep test audio files in root
- `!test_noise_conditions.png` - Keep test visualization
- `!docs/images/*` - Keep documentation images
- `!requirements*.txt` - Keep all requirement files

### HuggingFace Cache
The HuggingFace cache can get very large. Consider setting:
```bash
export HF_HOME=/path/to/external/drive/.cache/huggingface
```

## Usage Tips

1. **Before committing large files**, check with:
   ```bash
   git status --porcelain | grep -E "^(A|M)" | awk '{print $2}' | xargs -I {} ls -lh {}
   ```

2. **If you accidentally commit a large file**:
   ```bash
   git rm --cached large_file.bin
   git commit -m "Remove large file"
   ```

3. **To clean up ignored files**:
   ```bash
   git clean -fdX  # Remove ignored files
   git clean -fdx  # Remove all untracked files (careful!)
   ``` 