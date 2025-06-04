# Morse Code Whisper Evaluation Report

## Overall Performance
- **Word Error Rate (WER)**: 166.67%
- **Character Error Rate (CER)**: 567.61%
- **Noise Sample Accuracy**: 0.00%
- **Number of samples**: 7
  - Morse samples: 6
  - Noise samples: 1

## Chunking Analysis
- **Non-chunked audio (<30s)**: 7 samples, avg WER: 300.00%
- **Chunked audio (>30s)**: 0 samples, avg WER: 0.00%

### Performance by Duration
- **0-30s**: 300.00% (n=7)

## Performance by Contest Type
- **IARU**: 100.00% (n=2)
- **NAQP**: 800.00% (n=1)
- **ARRLDX**: 100.00% (n=1)
- **CQWPX**: 450.00% (n=2)
- **NOISE**: 100.00% (n=1)

## Performance by Signal Condition
- **clean**: 450.00% (n=2)
- **contest_moderate**: 100.00% (n=1)
- **contest_poor**: 100.00% (n=1)
- **dx_expedition**: 450.00% (n=2)
- **contest_good**: 100.00% (n=1)

## Performance by Speed (WPM)
- **0-4 WPM**: 100.00% (n=1)
- **15-19 WPM**: 100.00% (n=1)
- **20-24 WPM**: 800.00% (n=1)
- **25-29 WPM**: 450.00% (n=2)
- **30-34 WPM**: 100.00% (n=1)
- **35-39 WPM**: 100.00% (n=1)

## Most Common Errors
### Substitutions (Top 20)
- `CQ → ლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლ�`: 2 times
- `CQ → වවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවව`: 1 times
- `I7? → I'll`: 1 times
- `NL3ND → ლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლ�`: 1 times
- `73 → I'll`: 1 times

### Sample Callsign Errors
- `I7?` → `I'll`
- `NL3ND` → `ლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლ�`

## Perfect Transcriptions: 0/7 (0.0%)

## Worst Performing Samples

### Sample 1 (WER: 800.00%)
- **Reference**: `I7?`
- **Prediction**: `I'll see you in the next video. Bye.`
- **Conditions**: 29 WPM, clean, duration: 4.0s

### Sample 2 (WER: 800.00%)
- **Reference**: `73`
- **Prediction**: `I'll see you in the next video. Bye!`
- **Conditions**: 21 WPM, dx_expedition, duration: 3.7s

### Sample 3 (WER: 100.00%)
- **Reference**: `CQ DE NS9D K NS9D DE IK2ES 599 9 TU`
- **Prediction**: ` වවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවවව`
- **Conditions**: 37 WPM, clean, duration: 15.6s

### Sample 4 (WER: 100.00%)
- **Reference**: `NL3ND 599 KW`
- **Prediction**: ` ლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლ�`
- **Conditions**: 27 WPM, contest_moderate, duration: 8.8s

### Sample 5 (WER: 100.00%)
- **Reference**: `CQ DE PR0DI5 K`
- **Prediction**: ` ლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლ�`
- **Conditions**: 31 WPM, contest_poor, duration: 8.7s

### Sample 6 (WER: 100.00%)
- **Reference**: `CQ DE`
- **Prediction**: ` ლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლლ�`
- **Conditions**: 16 WPM, dx_expedition, duration: 5.4s

### Sample 7 (WER: 100.00%)
- **Reference**: ``
- **Prediction**: ` ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ ლ �`
- **Conditions**: 0 WPM, contest_good, duration: 6.8s
