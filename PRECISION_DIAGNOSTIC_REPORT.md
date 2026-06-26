# Precision Diagnostic Report

This note investigates why Stage 2 alert precision is modest in absolute terms even when the method outperforms WSTS-as-alerter and naive 4 km flooding.

## Short Answer

The main reason is that the alerting task is extremely sparse. In the 2020 candidate universe, only about 4.9% of candidate alert locations are positives. A precision around 0.25 is therefore not "only 25%" in isolation; it is about 5x the raw candidate base rate at the deployed alert budgets, while still issuing enough alerts to catch meaningful fire spread.

The precision is also budget-dependent. If the system is allowed to send only the very highest-confidence alerts, precision becomes much higher. As the alert budget expands, the system intentionally includes more borderline locations to improve recall and event-day catch rate, so precision falls.

## Diagnostic Method

The diagnostic evaluates precision sensitivity on the 2020 scored candidate files:

- `outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_full2020_retrained_stage2/stage2_2020_scored_candidates.csv`
- Original-checkpoint full-2020 scored candidate CSV.

For each run, precision is measured at fixed global alert budgets. This shows how precision changes as the alert budget changes. The matched-volume comparison remains the primary alert-quality result because it uses the frozen Stage 2 threshold and one global WSTS threshold matched to that run's Stage 2 alert volume.

Diagnostic output:

- `outputs/stage2/precision_diagnostics/precision_budget_diagnostics.json`

## Candidate Base Rate

| Run | Candidate rows | Positive candidates | Candidate positive rate |
|---|---:|---:|---:|
| Regenerated-checkpoint support | 5,212,827 | 255,105 | 0.0489 |
| original-checkpoint rerun | 5,212,363 | 255,761 | 0.0491 |

This means a random candidate alert would only be correct about 4.9% of the time.

## High-Confidence Alert Region

This table focuses on the most confident alert budgets. It is the clearest table for showing Professor how precision behaves when only a small number of alerts are issued per event-day.

### Regenerated-checkpoint support run

| Alerts/day | Stage 2 precision | Stage 2 recall | WSTS precision | WSTS recall | Stage 2 lift over base |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.6963 | 0.0298 | 0.5663 | 0.0242 | 14.2x |
| 10 | 0.6480 | 0.0555 | 0.5251 | 0.0450 | 13.2x |
| 15 | 0.5892 | 0.0757 | 0.4924 | 0.0632 | 12.0x |
| 20 | 0.5281 | 0.0904 | 0.4601 | 0.0788 | 10.8x |
| 25 | 0.4880 | 0.1044 | 0.4362 | 0.0934 | 10.0x |

### original-checkpoint rerun

| Alerts/day | Stage 2 precision | Stage 2 recall | WSTS precision | WSTS recall | Stage 2 lift over base |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.6675 | 0.0285 | 0.5740 | 0.0245 | 13.6x |
| 10 | 0.6331 | 0.0541 | 0.5142 | 0.0439 | 12.9x |
| 15 | 0.5934 | 0.0760 | 0.4751 | 0.0609 | 12.1x |
| 20 | 0.5604 | 0.0957 | 0.4459 | 0.0762 | 11.4x |
| 25 | 0.5271 | 0.1125 | 0.4238 | 0.0905 | 10.7x |

## Broader Operating Range

### Regenerated-checkpoint support run

| Alerts/day | Stage 2 precision | Stage 2 recall | WSTS precision | WSTS recall | Stage 2 lift over base | WSTS lift over base |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.6963 | 0.0298 | 0.5663 | 0.0242 | 14.2x | 11.6x |
| 10 | 0.6480 | 0.0555 | 0.5251 | 0.0450 | 13.2x | 10.7x |
| 15 | 0.5892 | 0.0757 | 0.4924 | 0.0632 | 12.0x | 10.1x |
| 20 | 0.5281 | 0.0904 | 0.4601 | 0.0788 | 10.8x | 9.4x |
| 25 | 0.4880 | 0.1044 | 0.4362 | 0.0934 | 10.0x | 8.9x |
| 50 | 0.3866 | 0.1655 | 0.3512 | 0.1503 | 7.9x | 7.2x |
| 75 | 0.3399 | 0.2183 | 0.2973 | 0.1909 | 6.9x | 6.1x |
| 90 | 0.3203 | 0.2468 | 0.2762 | 0.2128 | 6.5x | 5.6x |
| 114 | 0.2965 | 0.2893 | 0.2515 | 0.2455 | 6.1x | 5.1x |
| 150 | 0.2676 | 0.3436 | 0.2243 | 0.2881 | 5.5x | 4.6x |
| 250 | 0.2201 | 0.4712 | 0.1735 | 0.3713 | 4.5x | 3.5x |

### original-checkpoint rerun

| Alerts/day | Stage 2 precision | Stage 2 recall | WSTS precision | WSTS recall | Stage 2 lift over base | WSTS lift over base |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.6675 | 0.0285 | 0.5740 | 0.0245 | 13.6x | 11.7x |
| 10 | 0.6331 | 0.0541 | 0.5142 | 0.0439 | 12.9x | 10.5x |
| 15 | 0.5934 | 0.0760 | 0.4751 | 0.0609 | 12.1x | 9.7x |
| 20 | 0.5604 | 0.0957 | 0.4459 | 0.0762 | 11.4x | 9.1x |
| 25 | 0.5271 | 0.1125 | 0.4238 | 0.0905 | 10.7x | 8.6x |
| 50 | 0.4052 | 0.1730 | 0.3503 | 0.1496 | 8.3x | 7.1x |
| 75 | 0.3489 | 0.2234 | 0.2995 | 0.1918 | 7.1x | 6.1x |
| 90 | 0.3302 | 0.2537 | 0.2753 | 0.2115 | 6.7x | 5.6x |
| 114 | 0.3065 | 0.2983 | 0.2503 | 0.2436 | 6.2x | 5.1x |
| 150 | 0.2781 | 0.3562 | 0.2300 | 0.2946 | 5.7x | 4.7x |
| 250 | 0.2278 | 0.4863 | 0.1847 | 0.3942 | 4.6x | 3.8x |

## Interpretation

The low-looking precision is not best explained as "the model is weak." The diagnostic shows that Stage 2 is strongly enriching positives compared with the underlying candidate population:

- The candidate base rate is only about 0.049.
- At strict budgets of 5 to 10 alerts/day, Stage 2 precision is roughly 0.63 to 0.70.
- Around larger operational budgets, precision drops because the system is trading some precision for greater recall and event-day coverage.
- Across the tested budgets, Stage 2 stays ahead of WSTS precision in both validation runs.

So the justification is:

Stage 2 operates in a rare-event spatial alerting setting where the raw positive rate is about 5%. Its precision is modest in absolute terms because the event is sparse and the selected alert budget is designed to catch more spread, not only the easiest few pixels. Relative to the base rate, WSTS, and naive flooding, Stage 2 is materially better.

## Suggested Thesis Framing

Do not frame the result as "Stage 2 is perfect" or "every alert should be trusted equally." A more defensible statement is:

> Although absolute precision remains limited by the rarity and spatial ambiguity of wildfire spread, Stage 2 substantially enriches true positives relative to the candidate base rate and outperforms WSTS-as-alerter at matched alert volume. This means the model is useful as a prioritization and alert-screening layer, not as a guarantee that every alerted pixel will burn.
