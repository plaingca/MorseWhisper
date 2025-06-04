# Morse Code Whisper Evaluation Report

## Overall Performance
- **Word Error Rate (WER)**: 93.65%
- **Character Error Rate (CER)**: 86.30%
- **Number of samples**: 104

## Chunking Analysis
- **Non-chunked audio (<30s)**: 54 samples, avg WER: 97.33%
- **Chunked audio (>30s)**: 50 samples, avg WER: 93.30%

### Performance by Duration
- **0-30s**: 97.33% (n=54)
- **30-60s**: 92.89% (n=33)
- **60-90s**: 93.97% (n=16)
- **90-∞s**: 96.30% (n=1)

## Performance by Contest Type
- **CQWPX**: 93.47% (n=17)
- **NAQP**: 100.82% (n=17)
- **CQWW**: 94.58% (n=16)
- **ARRLDX**: 98.17% (n=19)
- **SPRINT**: 92.99% (n=14)
- **FIELD_DAY**: 92.82% (n=11)
- **IARU**: 90.82% (n=7)
- **SWEEPSTAKES**: 93.67% (n=3)

## Performance by Signal Condition
- **contest_good**: 91.71% (n=19)
- **dx_expedition**: 98.84% (n=19)
- **contest_moderate**: 95.64% (n=24)
- **contest_poor**: 93.87% (n=21)
- **clean**: 96.86% (n=21)

## Performance by Speed (WPM)
- **15-19 WPM**: 98.57% (n=13)
- **20-24 WPM**: 93.87% (n=16)
- **25-29 WPM**: 93.38% (n=20)
- **30-34 WPM**: 97.07% (n=27)
- **35-39 WPM**: 93.65% (n=23)
- **40-44 WPM**: 99.06% (n=5)

## Most Common Errors
### Substitutions (Top 20)
- `TEST → CQ`: 25 times
- `CQ → DE`: 22 times
- `CQ → ?`: 18 times
- `DE → ?`: 14 times
- `599 → DE`: 12 times
- `TEST → ?`: 12 times
- `QRZ? → ?`: 11 times
- `599 → ?`: 8 times
- `K → DE`: 8 times
- `DE → VE6JN`: 8 times
- `DE → JS?`: 7 times
- `DE → QRZ?`: 7 times
- `599 → QRZ?`: 7 times
- `GI? → ?`: 7 times
- `TU → DE`: 6 times
- `K → CQ`: 6 times
- `TEST → DE`: 6 times
- `TEST → VE6JN`: 5 times
- `K → QRZ?`: 5 times
- `599 → CQ`: 5 times

### Sample Callsign Errors
- `JS1V4` → `?`
- `4T909` → `?`
- `JS1V4` → `?`
- `JS1V4` → `?`
- `4T909` → `AGNG`
- `EE4TF` → `CQ`
- `EE4TF` → `DE`
- `EE4TF` → `VE6JN`
- `IK8YUD` → `VE6JN`
- `GJ4AOH` → `GJ5WG`

## Perfect Transcriptions: 0/104 (0.0%)

## Worst Performing Samples

### Sample 1 (WER: 122.22%)
- **Reference**: `CQ CQ CQ DE DE5ZO9 DE5ZO9 DE5ZO9 DE5ZO9 F9FBL PAT AR GEORGE DC GL QSL TU NAQP F9FBL`
- **Prediction**: `… … … … … … … … … … … … CQ CQ DE GJ9GWN GJ9GWN GJ9GWN GJ9GWN GJ9GWN DQ? QSL`
- **Conditions**: 15 WPM, contest_poor, duration: 81.3s
- **Note**: Audio was chunked (>30s)

### Sample 2 (WER: 120.00%)
- **Reference**: `CQ CQ CQ DE VO5WM5 VO5WM5 VO5WM5 VO5WM5 DE VE6YC 599 1500 K TU 599 1500 VO5WM5 TU QRZ? VE6YC`
- **Prediction**: `I? QRZ? QRZ? DE KS? DE KS? QRZ? DE KS? DE KS? DE KS? DE KS? DE KS? DE KS? DE KS? DE KS? DE KS?`
- **Conditions**: 35 WPM, contest_moderate, duration: 41.7s
- **Note**: Audio was chunked (>30s)

### Sample 3 (WER: 115.00%)
- **Reference**: `CQ TEST CQ TEST DE DB9QH DB9QH K DB9QH DE NR4P 599 32 K 599 15 GL TU QRZ? NR4P`
- **Prediction**: `CQ CQ DE JI? JI? JI? QRZ? QRZ? DE JI? DE JI? DE JI? DE JI? DE JI? DE JI? DE JI? ? ? ? ?`
- **Conditions**: 34 WPM, contest_moderate, duration: 34.7s
- **Note**: Audio was chunked (>30s)

### Sample 4 (WER: 113.04%)
- **Reference**: `CQ TEST CQ TEST DE I7VQ I7VQ K I7VQ WN2H 599 RI CALL? I7VQ WN2H 599 RI 599 1500 GL TU QRZ? WN2H`
- **Prediction**: `CQ CQ DE JS? JS? QRZ? QRZ? DE JS? DE JS? DE JS? QRZ? DE JS? DE JS? DE JS? QQ TEST? QQ TEST? QQ TEST? QQ TEST?`
- **Conditions**: 32 WPM, dx_expedition, duration: 45.3s
- **Note**: Audio was chunked (>30s)

### Sample 5 (WER: 107.32%)
- **Reference**: `TEST NZ7E NZ7E TEST NZ7E JACK WV RH9XM6 CALL? NZ7E JACK WV RH9XM6 R JANE WA TU TU QRZ? RH9XM6 TEST NZ7E NZ7E TEST NZ7E JACK WV RH9XM6 CALL? NZ7E JACK WV RH9XM6 R JANE WA TU TU QRZ? RH9XM6 QRZ?`
- **Prediction**: `DQ? QRZ? QRZ? AGN DE DQ? QRZ? DE DQ? DE DQ? DE DQ? AGN DE DQ? QRZ? DE DQ? AGN DE DQ? DE DQ? CQ CQ DE GJ9GWG GJ9GWG GJ9GWG GJ9GWG GJ9GWG DE GJ9GWG DE GJ9GWG DE GJ9GWG AGN GJ9GWG GJ9GWG DE GJ9GWG GJ9GWG`
- **Conditions**: 40 WPM, contest_good, duration: 68.2s
- **Note**: Audio was chunked (>30s)

### Sample 6 (WER: 105.26%)
- **Reference**: `TEST EC7OS5 EC7OS5 TEST EC7OS5 DE KU7BIP 1802 CARL CO K 1259 RICK ME GL QSL TU TEST KU7BIP`
- **Prediction**: `CQ CQ DE JI? JI? JI? DE JI? QRZ? QRZ? JI? DE JI? DE JI? DE JI? DE JI? DE JI?`
- **Conditions**: 35 WPM, contest_poor, duration: 35.5s
- **Note**: Audio was chunked (>30s)

### Sample 7 (WER: 105.13%)
- **Reference**: `CQ TEST CQ TEST DE DI2BV DI2BV K DI2BV JS4RA 599 100 599 KW GL QSL TU TEST JS4RA CQ TEST CQ TEST DE DI2BV DI2BV K DI2BV JS4RA 599 100 599 KW GL QSL TU TEST JS4RA QRZ?`
- **Prediction**: `DQ1F? DQ1F? DQ1F? QRZ? QRZ? DE DQ1F? QRZ? DE DQ1F? QRZ? DE DQ1F? CQ CQ DE IU? IU? QRZ? CQ DE IU? IU? AGN CQ DE IU? AGN CQ DE IU? AGN CQ DE IU? AGN CQ DE IU? ? ? ? ?`
- **Conditions**: 40 WPM, dx_expedition, duration: 61.1s
- **Note**: Audio was chunked (>30s)

### Sample 8 (WER: 100.00%)
- **Reference**: `TEST EE4TF EE4TF TEST EE4TF IK8YUD BEN AL R PAUL WA TU TU QRZ? IK8YUD`
- **Prediction**: `CQ CQ DE VE6JN VE6JN VE6JN DE`
- **Conditions**: 24 WPM, dx_expedition, duration: 35.7s
- **Note**: Audio was chunked (>30s)

### Sample 9 (WER: 100.00%)
- **Reference**: `TEST RZ5NCG RZ5NCG TEST RZ5NCG I5YSJ 599 100 CALL? RZ5NCG I5YSJ 599 100 599 KW GL 73 GL I5YSJ TEST`
- **Prediction**: `I? QRZ? QRZ? I? DE TEST? EG8JWN EG8JWN K R`
- **Conditions**: 28 WPM, contest_moderate, duration: 53.4s
- **Note**: Audio was chunked (>30s)

### Sample 10 (WER: 100.00%)
- **Reference**: `TEST JR0RQA JR0RQA TEST JR0RQA 4687 TOM NM DG2UI R 3306 ANN TN TU TU QRZ? DG2UI`
- **Prediction**: `CQ CQ DE GJ9ZWG GJ9ZWG GJ9ZWG GJ9ZWG GJ9ZWG 569 QSL`
- **Conditions**: 30 WPM, clean, duration: 37.9s
- **Note**: Audio was chunked (>30s)
