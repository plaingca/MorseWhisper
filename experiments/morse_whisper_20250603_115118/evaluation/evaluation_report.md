# Morse Code Whisper Evaluation Report

## Overall Performance
- **Word Error Rate (WER)**: 82.13%
- **Character Error Rate (CER)**: 106.99%
- **Number of samples**: 500

## Performance by Contest Type
- **FIELD_DAY**: 86.64% (n=73)
- **CQWPX**: 97.78% (n=69)
- **IARU**: 66.13% (n=59)
- **CQWW**: 183.06% (n=71)
- **SPRINT**: 81.58% (n=59)
- **NAQP**: 87.50% (n=55)
- **SWEEPSTAKES**: 81.01% (n=56)
- **ARRLDX**: 68.08% (n=58)

## Performance by Signal Condition
- **clean**: 93.37% (n=100)
- **contest_poor**: 73.36% (n=98)
- **dx_expedition**: 93.42% (n=100)
- **contest_moderate**: 142.39% (n=106)
- **contest_good**: 74.17% (n=96)

## Performance by Speed (WPM)
- **15-19 WPM**: 98.56% (n=94)
- **20-24 WPM**: 86.29% (n=98)
- **25-29 WPM**: 157.97% (n=99)
- **30-34 WPM**: 70.75% (n=93)
- **35-39 WPM**: 69.24% (n=98)
- **40-44 WPM**: 75.33% (n=18)

## Most Common Errors
### Substitutions (Top 20)
- `TEST → DE`: 40 times
- `TEST → CQ`: 34 times
- `599 → ?`: 27 times
- `TU → TEST`: 15 times
- `QSL → TU`: 12 times
- `TU → QSL`: 12 times
- `GL → TU`: 10 times
- `TU → 599`: 9 times
- `73 → GL`: 6 times
- `CQ → TX`: 6 times
- `CQ → CQWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW`: 6 times
- `599 → TU`: 5 times
- `K → KW`: 5 times
- `599 → 1500`: 5 times
- `K → TU`: 5 times
- `TU → KW`: 5 times
- `R → 599`: 5 times
- `TU → GL`: 5 times
- `RJ5MC → RY5MG`: 5 times
- `GL → TEST`: 5 times

### Sample Callsign Errors
- `VA0DB` → `DE`
- `IU9EHB` → `4A`
- `4A` → `SD`
- `4A` → `QSL`
- `IU9EHB` → `4A`
- `JF2XZU` → `TEST`
- `WR0LRN` → `WW0LRN`
- `WR0LRN` → `TEST`
- `WR0LRN` → `WW4T`
- `WW4T` → `599`

## Perfect Transcriptions: 18/500 (3.6%)

## Worst Performing Samples

### Sample 1 (WER: 7500.00%)
- **Reference**: `EC?`
- **Prediction**: `? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG? EG`
- **Conditions**: 28 WPM, contest_moderate

### Sample 2 (WER: 1022.22%)
- **Reference**: `CQ CQ CQ DE DB3LLN DB3LLN DB3LLN DB3LLN 599 9225 DA3HB 569 322 GL QSL TU TEST DA3HB`
- **Prediction**: `CQ CQ CQ DE DB3LN DB3LN DB3LN DB3LN 599 KW GL QSL QSL GL QSL GL QSL GL QSL GL QSL GL QSL GL QSL GL QSL GL QSL GL QSL GL QSL GL QSL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL`
- **Conditions**: 18 WPM, clean

### Sample 3 (WER: 852.63%)
- **Reference**: `CQ CQ CQ DE EA6JBF EA6JBF EA6JBF EA6JBF JP6DX 2B ME TU 2A SC EA6JBF QSL TU TEST JP6DX`
- **Prediction**: `CQ CQ CQ DE EA6JBF EA6JBF EA6JBF EA6JBF EA6JBF JP6DX 2B ME WI? JP6DX 2B ME WI? JP6DX 2B ME WI? JP6DX 2B ME WI WI GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL`
- **Conditions**: 25 WPM, clean

### Sample 4 (WER: 840.00%)
- **Reference**: `CQ CQ CQ DE M32FV M32FV M32FV M32FV DE VK1PBPM 599 6679 K TU 599 9714 M32FV TU QRZ? VK1PBPM`
- **Prediction**: `GJ9 GJ9 GJ9 DE M3FV GJ9 M3FV GJ9 AGN M3FV GJ9 AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN AGN GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL GL`
- **Conditions**: 16 WPM, dx_expedition

### Sample 5 (WER: 533.33%)
- **Reference**: `LO5NAR 599 59`
- **Prediction**: ` 599 599 599 599 599 599 599 599 599 599 GL 73 GL GL QSL TU TEST                                                                                                                                                                                                `
- **Conditions**: 22 WPM, contest_good

### Sample 6 (WER: 450.00%)
- **Reference**: `TEST VY2XIF VY2XIF TEST`
- **Prediction**: ` 599 599 599 599 599 599 599 599 599 599 599 599 599 599 599 GL QSL TU TEST                                                                                                                                                                                         `
- **Conditions**: 24 WPM, contest_poor

### Sample 7 (WER: 375.00%)
- **Reference**: `CQ TEST CQ TEST DE RG8WK RG8WK K RG8WK UB7HD RICK ME TU GEORGE MN RG8WK QSL TU TEST UB7HD`
- **Prediction**: `CQ CQ CQ DE RG8WK RG8WK RG8WK RG8WK UB7HD 599 KW RG8WK TU 599 KW RG8WK TU 599 KW UB7HD 599 KW RG8WK TU 599 KW RG8WK TU 599 KW UB7HD 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW RG8WK TU 599 KW`
- **Conditions**: 21 WPM, contest_good

### Sample 8 (WER: 366.67%)
- **Reference**: `RP6QWS 599 8940`
- **Prediction**: ` 890? 890? 890? 890? 890? 890? 599 8 GL QSL TU TEST                                                                                                                                                                                                  `
- **Conditions**: 27 WPM, dx_expedition

### Sample 9 (WER: 360.00%)
- **Reference**: `JK0MQ 9095 B 54 AL`
- **Prediction**: ` 999 599 599 599 599 599 599 599 599 599 599 599 599 599 GL QSL TU TEST                                                                                                                                                                                           `
- **Conditions**: 16 WPM, dx_expedition

### Sample 10 (WER: 300.00%)
- **Reference**: `WA2L?`
- **Prediction**: `? ? ?                                                                                                                                                                                                                           `
- **Conditions**: 33 WPM, contest_poor
