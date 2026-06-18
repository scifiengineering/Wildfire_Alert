# Confidence-Threshold Operating Points

This report uses fixed confidence thresholds as the operating point. For each threshold, candidate locations with scores greater than or equal to that threshold are alerted, then alerts/day, precision, recall, F1, and true-positive alerts are measured.

- Candidate rows: 5,212,827
- Positive candidate rows: 255,105
- Candidate positive rate: 0.0489
- Event-days: 2,184

## Stage 2 High-Confidence Region

| Confidence threshold | Alerts/day | Precision | Recall | F1 | True-positive alerts | Lift over base |
|---:|---:|---:|---:|---:|---:|---:|
| 0.950 | 2.92 | 0.7188 | 0.0179 | 0.0350 | 4,577 | 14.7x |
| 0.940 | 5.70 | 0.6876 | 0.0335 | 0.0639 | 8,553 | 14.1x |
| 0.930 | 7.99 | 0.6656 | 0.0455 | 0.0853 | 11,620 | 13.6x |
| 0.920 | 9.58 | 0.6527 | 0.0536 | 0.0990 | 13,661 | 13.3x |
| 0.910 | 11.48 | 0.6313 | 0.0620 | 0.1130 | 15,824 | 12.9x |
| 0.900 | 13.74 | 0.6045 | 0.0711 | 0.1273 | 18,142 | 12.4x |
| 0.875 | 25.44 | 0.4845 | 0.1055 | 0.1733 | 26,918 | 9.9x |
| 0.850 | 45.84 | 0.3968 | 0.1557 | 0.2236 | 39,720 | 8.1x |

## Stage 2 Broader Threshold Range

| Confidence threshold | Alerts/day | Precision | Recall | F1 | True-positive alerts | Lift over base |
|---:|---:|---:|---:|---:|---:|---:|
| 0.825 | 69.13 | 0.3489 | 0.2065 | 0.2594 | 52,673 | 7.1x |
| 0.800 | 91.14 | 0.3190 | 0.2489 | 0.2796 | 63,493 | 6.5x |
| 0.775 | 109.18 | 0.3006 | 0.2810 | 0.2905 | 71,687 | 6.1x |
| 0.750 | 125.31 | 0.2870 | 0.3079 | 0.2971 | 78,541 | 5.9x |
| 0.700 | 158.41 | 0.2617 | 0.3550 | 0.3013 | 90,551 | 5.3x |
| 0.650 | 195.14 | 0.2416 | 0.4036 | 0.3023 | 102,972 | 4.9x |
| 0.600 | 238.19 | 0.2246 | 0.4579 | 0.3013 | 116,816 | 4.6x |
| 0.550 | 283.40 | 0.2079 | 0.5045 | 0.2945 | 128,708 | 4.2x |
| 0.500 | 330.48 | 0.1927 | 0.5451 | 0.2847 | 139,053 | 3.9x |

## WSTS Score Threshold Context

This table applies the same numeric thresholds to the WSTS/Stage-1 probability score. It is useful context, but it should not be treated as a calibrated confidence equality between models.

| Score threshold | Alerts/day | Precision | Recall | F1 | True-positive alerts | Lift over base |
|---:|---:|---:|---:|---:|---:|---:|
| 0.950 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0x |
| 0.940 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0x |
| 0.930 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0x |
| 0.920 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0x |
| 0.910 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0x |
| 0.900 | 0.24 | 0.6187 | 0.0012 | 0.0025 | 318 | 12.6x |
| 0.875 | 4.25 | 0.5791 | 0.0211 | 0.0407 | 5,378 | 11.8x |
| 0.850 | 11.11 | 0.5183 | 0.0493 | 0.0900 | 12,573 | 10.6x |
| 0.825 | 18.87 | 0.4664 | 0.0754 | 0.1298 | 19,224 | 9.5x |
| 0.800 | 27.11 | 0.4265 | 0.0990 | 0.1607 | 25,255 | 8.7x |
| 0.775 | 35.99 | 0.3912 | 0.1206 | 0.1843 | 30,756 | 8.0x |
| 0.750 | 45.03 | 0.3646 | 0.1406 | 0.2029 | 35,862 | 7.5x |
| 0.700 | 62.20 | 0.3210 | 0.1709 | 0.2230 | 43,598 | 6.6x |
| 0.650 | 77.15 | 0.2939 | 0.1941 | 0.2338 | 49,520 | 6.0x |
| 0.600 | 92.56 | 0.2730 | 0.2163 | 0.2413 | 55,177 | 5.6x |
| 0.550 | 110.72 | 0.2542 | 0.2409 | 0.2474 | 61,464 | 5.2x |
| 0.500 | 136.51 | 0.2339 | 0.2734 | 0.2521 | 69,750 | 4.8x |

## Interpretation

This confidence-threshold view answers the professor's possible alternative framing directly. Instead of choosing 5, 10, or 25 alerts/day first, we choose a confidence threshold first and observe how many alerts/day the system emits.

The pattern is consistent with the alert-budget view: stricter confidence thresholds produce fewer alerts/day and higher precision, while relaxed thresholds capture more true positives and increase recall at the cost of precision.
