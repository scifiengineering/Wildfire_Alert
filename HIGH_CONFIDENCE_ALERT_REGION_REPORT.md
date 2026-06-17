# High-Confidence Alert Region

This table focuses only on the most confident alert operating points. It is meant as a compact view for showing how precision behaves when the system issues a small number of alerts per event-day.

`Alerts/day` is the alert budget after ranking candidate locations by confidence. The `Stage 2 min confidence` column shows the implied confidence cutoff: fewer alerts/day means a stricter cutoff, so only higher-confidence locations are alerted.

| Alerts/day | Stage 2 min confidence | Stage 2 precision | WSTS precision | Stage 2 lift over base |
|---:|---:|---:|---:|---:|
| 5 | 0.9425 | 0.6963 | 0.5663 | 14.2x |
| 10 | 0.9173 | 0.6480 | 0.5251 | 13.2x |
| 15 | 0.8958 | 0.5892 | 0.4924 | 12.0x |
| 20 | 0.8841 | 0.5281 | 0.4601 | 10.8x |
| 25 | 0.8757 | 0.4880 | 0.4362 | 10.0x |

Interpretation: the confidence cutoff decreases as more alerts/day are allowed. This is why precision is highest at 5 to 10 alerts/day and then gradually decreases as the system includes less certain candidate locations. The alert count itself is not the cause; it is the result of relaxing the confidence threshold.

