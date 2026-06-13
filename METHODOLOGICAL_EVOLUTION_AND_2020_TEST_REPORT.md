# Methodological Evolution and 2020 External Test Report

## 1. Purpose of This Report

This report explains how the thesis methodology evolved from an initial 2019
proof of concept into a more defensible wildfire alerting experiment. The two
major corrections were:

1. replacing overlapping two-way event splits with **three leakage-free folds**;
2. replacing a single next-day alert demonstration with a **rolling three-day
   alert protocol**, followed by a frozen evaluation on the unseen 2020 season.

The final question is no longer simply:

> Can a model predict tomorrow's active-fire pixels?

It is:

> Can a model repeatedly identify the most relevant nearby alert locations over
> a three-day horizon, remove locations that no longer justify an alert, and
> reduce alert volume without losing useful spatial discrimination?

---

## 2. Methodological Evolution at a Glance

| Phase | Data separation | Temporal evaluation | Main purpose |
|---|---|---|---|
| Initial prototype | Two independently generated 2019 event splits | One-day prediction | Demonstrate that Stage 2 could improve candidate ranking |
| Corrected 2019 experiment | Three mutually exclusive event folds | Rolling Day 6, Day 7 and Day 8 alerts | Remove event leakage and test alert persistence |
| Final external evaluation | Train and tune on 2019; test on 2020 | Same frozen rolling protocol | Test generalization to an unseen fire season |

The initial prototype was useful for developing the two-stage architecture, but
its numerical results are not the final thesis results. The authoritative
methodology is the leakage-free three-fold 2019 experiment plus the frozen 2020
external test.

---

## 3. Core Two-Stage Alerting Framework

The final methodology retains the original two-stage intuition.

### Stage 1: Spatial spread prediction

Stage 1 is a U-Net-based model. It receives five consecutive daily WSTS raster
observations and produces a pixel-level probability map for the following day.
Each daily raster contains active-fire information together with environmental
variables such as vegetation, weather, forecast weather, terrain and drought.

The purpose of Stage 1 is not to make the final alert decision. Its role is to
provide a learned spatial signal answering:

> Which pixels look compatible with the next spread pattern?

### Stage 2: Location-level discrimination

Stage 2 converts the dense raster problem into a candidate-ranking problem.
Candidate pixels are described using:

- Stage 1 probability and top-risk indicators;
- distance and direction from the current fire;
- wind and forecast-wind alignment;
- temperature, precipitation and humidity;
- slope, aspect and elevation;
- vegetation indices and land-cover information;
- drought and fire-energy indicators.

A class-weighted LightGBM classifier then assigns an alert score to every
candidate location. This stage answers:

> Among the many locations near the fire, which ones deserve the limited alert
> capacity?

This distinction is central. Stage 1 predicts a map, while Stage 2 makes the map
operationally selective.

---

## 4. Major Change One: From Two Overlapping Splits to Three Leakage-Free Folds

### 4.1 Problem in the initial evaluation

The initial prototype used two independently generated event splits. Although
each individual split separated its own training and validation events, the two
held-out lists were not coordinated. Several fire IDs consequently appeared in
both held-out directions.

This did not necessarily place the same fire in the training and validation
partition of one individual run. However, it invalidated the interpretation
that the two directions represented clean complementary folds. It also made the
combined result difficult to describe as proper cross-validation.

### 4.2 Corrected intuition

The unit of independence must be the **entire fire event**, not a raster pixel
or event-day.

Observations from one fire are strongly related across time. If days or pixels
from the same fire appear on both sides of a split, the model can benefit from
event-specific spatial and environmental patterns. A convincing evaluation
therefore holds out complete fire IDs.

The corrected approach partitions every eligible 2019 event into exactly one of
three validation groups. The groups are disjoint:

```text
Fold 0 validation ∩ Fold 1 validation = empty
Fold 0 validation ∩ Fold 2 validation = empty
Fold 1 validation ∩ Fold 2 validation = empty
```

The generated split manifest explicitly reports:

```text
has_validation_overlap: false
```

### 4.3 Final 2019 fold structure

The 2019 dataset contains 74 eligible fire events in this experiment.

| Held-out fold | Training events | Held-out events |
|---|---:|---:|
| Fold 0 | 49 | 25 |
| Fold 1 | 49 | 25 |
| Fold 2 | 50 | 24 |

Across the three directions, all 74 events receive one out-of-fold evaluation.
No event is held out in more than one fold.

### 4.4 Training direction

For Stage 1, each model is trained on the events outside its held-out fold:

```text
Stage 1 model 0: train on Folds 1 + 2, validate on Fold 0
Stage 1 model 1: train on Folds 0 + 2, validate on Fold 1
Stage 1 model 2: train on Folds 0 + 1, validate on Fold 2
```

Stage 2 follows the same held-out logic. For example, the holdout-0 LightGBM
model is trained using candidate rows from held-out predictions belonging to
Folds 1 and 2, then evaluated on Fold 0 candidates.

This is preferable to fitting Stage 2 on in-sample Stage 1 outputs because the
Stage 2 training rows are based on predictions made for events that were held
out from the corresponding Stage 1 model.

### 4.5 Corrected Stage 1 results

| Held-out fold | Best AP | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| Fold 0 | 0.0644 | 0.1224 | 0.0755 | 0.3227 |
| Fold 1 | 0.1158 | 0.1441 | 0.0866 | 0.4293 |
| Fold 2 | 0.1362 | 0.2690 | 0.2455 | 0.2973 |

These values confirm that dense spread segmentation remains difficult. The
important interpretation is not that Stage 1 solves wildfire spread alone.
Instead, it supplies a useful but imperfect risk feature to Stage 2.

### 4.6 Corrected Stage 2 results

| Held-out fold | Stage 2 AP | Within-event AUC | Best F1 |
|---|---:|---:|---:|
| Fold 0 | 0.2097 | 0.8631 | 0.2758 |
| Fold 1 | 0.2913 | 0.8935 | 0.3456 |
| Fold 2 | 0.2822 | 0.8724 | 0.3301 |

Stage 2 improved Average Precision over both distance-only and Stage-1-only
ranking in all three folds. Distance remained a strong feature and slightly
exceeded Stage 2 within-event AUC in two folds. The defensible conclusion is
therefore:

> Feature-engineered Stage 2 ranking improves precision-oriented candidate
> selection and alert efficiency, but distance remains a strong spatial
> baseline and is not dominated on every metric.

---

## 5. Major Change Two: From One-Day Alerts to a Rolling Three-Day Protocol

### 5.1 Original temporal formulation

The first formulation used five observed days to predict the next, sixth day:

```text
Days 1-5 observations -> Day 6 prediction
```

This is a valid next-day forecasting task, but it gives only a snapshot. An
operational alerting system must repeatedly reconsider locations as the fire and
environmental conditions evolve.

### 5.2 Corrected rolling formulation

The revised protocol creates three consecutive one-day-ahead decisions:

```text
Days 1-5 -> predict and alert for Day 6
Days 2-6 -> predict and alert for Day 7
Days 3-7 -> predict and alert for Day 8
```

The prediction horizon at every individual step is still **one day**. The
overall evaluation horizon is three days because the next-day model is applied
repeatedly.

This is preferable to making one direct three-day raster prediction because the
rolling approach can use newly observed conditions before producing the next
alert set.

### 5.3 What “learning from Day 6” means in the implemented system

The implementation uses the newly available Day 6 conditions when constructing
the Day 7 input window, and then uses Day 7 conditions for Day 8. It also
compares predictions with observed targets during evaluation.

However, the model weights are **not updated online** after Day 6 or Day 7.
There is no test-time retraining on the newly revealed ground truth. Therefore,
the accurate description is:

> Rolling prediction with updated observations and alert-state tracking.

It should not be described as online learning or continual model adaptation.

### 5.4 Expanding alert radius

The alert search radius expands with forecast progression:

| Rolling step | Target day | Search radius |
|---|---|---:|
| Step 0 | Day 6 | 2 km |
| Step 1 | Day 7 | 3 km |
| Step 2 | Day 8 | 4 km |

The intuition is that immediate spread should be searched more locally, while
later rolling steps allow a wider spatial region as the observed fire front
moves.

The underlying fair candidate table is generated from a 4 km ring plus Stage 1
top-risk cells, without injecting true future fire pixels. The evaluator then
restricts eligible locations to 2, 3 or 4 km according to the rolling step.

### 5.5 Alert-state tracking

For each event and rolling window, the evaluator maintains the previously
alerted raster coordinates.

At the next step, a location can be:

- **newly alerted** because it now exceeds the score threshold;
- **retained** because it remains above the threshold;
- **dropped** because it no longer qualifies.

The evaluator records the number of previous alerts removed, including previous
alerts originating inside the initial 2 km region. This converts “alert fatigue”
from a verbal claim into a measurable quantity.

### 5.6 Naive baselines

Two related naive comparisons are reported.

The daily naive comparison alerts every candidate within 4 km of the current
fire at each rolling step. It measures how much daily alert traffic is avoided
by discriminative scoring.

The fixed three-day baseline uses the active fire at the initial Day-5 decision
point, marks every location within 4 km, and evaluates those locations against
the final Day-8 target. This directly represents:

```text
Day-5 fire map -> naive 4 km alert region -> Day-8 outcome
```

The fixed baseline is the cleaner comparison for the final +3-day thesis claim.

---

## 6. 2019 Cross-Validated Rolling Results

Across all three held-out folds, the corrected 2019 rolling evaluation covered:

- 473 rolling windows;
- 1,419 event-day decisions;
- mean Stage 2 alerts: `37.13` per event-day;
- mean naive 4 km alerts: `1099.20` per event-day;
- model precision: `0.1266`;
- naive precision: `0.0269`;
- alert reduction: `97.12%`;
- rolling AP: `0.2644`;
- rolling AUC: `0.8414`.

The model therefore issued roughly one-thirtieth as many alerts while producing
substantially higher precision.

These are valid **2019 cross-validated held-out results**. They demonstrate
performance on unseen 2019 events, but they do not by themselves prove
generalization to another season.

---

## 7. Why the Methodology Moved to a 2020 Test

### 7.1 Limitation of 2019-only reporting

The 2019 subset did not contain a separate official test partition. Three-fold
event cross-validation solves event leakage and provides held-out estimates, but
model development, threshold selection and evaluation still occur within one
fire season.

A committee could reasonably ask:

> Does the method work on fires from a different year, or only on the
> environmental distribution represented in 2019?

### 7.2 Role assigned to each year

The corrected experimental roles are:

| Data | Methodological role |
|---|---|
| 2019 training folds | Learn Stage 1 and Stage 2 model parameters |
| 2019 held-out folds | Compare methods and select fixed operating thresholds |
| 2020 events | Independent temporal test of the frozen pipeline |

No Stage 1 or Stage 2 model was retrained using 2020. No feature was added after
observing 2020 performance. The alert radii and thresholds were fixed before
the 2020 evaluation.

This turns 2020 into an external temporal test rather than another validation
fold.

---

## 8. Step-by-Step 2020 External Test Pipeline

### Step 1: Convert 2020 GeoTIFF events to HDF5

All 201 events from 2020 were converted to event-wise HDF5 files using LZF
compression. A one-event smoke conversion was checked first for:

- expected raster dimensions and 23 channels;
- correct number of dates;
- LZF compression;
- preserved CRS and affine-transform metadata;
- successful loading into the five-day Stage 1 dataset.

The conversion changes storage and read efficiency only. It does not alter the
scientific content.

### Step 2: Apply all three frozen Stage 1 models

Each 2019 fold checkpoint was applied independently to every eligible 2020
event-day:

```text
2019 Stage 1 fold 0 model -> 2,184 predictions
2019 Stage 1 fold 1 model -> 2,184 predictions
2019 Stage 1 fold 2 model -> 2,184 predictions
```

This produces three probability estimates for each 2020 raster pixel.

### Step 3: Ensemble Stage 1

The three probabilities are averaged pixel by pixel:

```text
ensemble probability =
    (fold0 probability + fold1 probability + fold2 probability) / 3
```

The ensemble reduces dependence on one particular 2019 fold partition. The
Stage 1 top-risk fraction is frozen at `0.001`. A fixed ensemble threshold of
`0.7314453125`, calculated as the mean of the three 2019 fold thresholds, is
used for artifact compatibility and was not tuned on 2020.

### Step 4: Construct fair 2020 candidates

Candidates are generated from:

- a 4 km neighborhood around the observed current fire;
- Stage 1 top-risk pixels.

True future fire pixels are not inserted into the candidate set. This prevents
future-label leakage.

The resulting 2020 table contains:

- 201 events;
- 2,184 event-days;
- 5,212,367 candidate rows;
- 255,765 positive candidates;
- positive rate `4.91%`.

### Step 5: Apply the three frozen Stage 2 models

Every candidate is scored by all three 2019 LightGBM models. Their probabilities
are averaged:

```text
final Stage 2 score =
    (holdout0 model + holdout1 model + holdout2 model) / 3
```

This is a model ensemble, not retraining.

### Step 6: Apply the frozen Stage 2 threshold

The Stage 2 alert threshold was selected from 664,154 leakage-free 2019
out-of-fold candidate predictions:

```text
frozen threshold = 0.7777810642
```

At its 2019 pooled operating point:

- precision was `0.2947`;
- recall was `0.3434`;
- F1 was `0.3172`;
- alert rate was approximately `3.5%`.

The same numeric threshold was transferred to the averaged 2020 ensemble
scores. It was not adjusted after seeing 2020 labels.

One limitation should be acknowledged: the threshold was calibrated from pooled
out-of-fold single-model scores, while 2020 uses three-model averaged scores.
This preserves independence from 2020, but future work could reserve an
additional 2019 calibration partition specifically for ensemble-score
calibration.

### Step 7: Run rolling and fixed-baseline evaluation

The scored candidates are evaluated over:

- `1,794` rolling three-day windows;
- `5,382` event-day alert decisions;
- the `2 -> 3 -> 4 km` radius schedule;
- the frozen Stage 2 threshold;
- daily naive 4 km alerting;
- the fixed Day-5-to-Day-8 naive 4 km baseline.

---

## 9. Final 2020 Results

### 9.1 Rolling daily comparison

| Metric | Discriminative model | Daily naive 4 km |
|---|---:|---:|
| Mean alerts per event-day | 89.53 | 2589.46 |
| Mean precision | 0.1747 | 0.0396 |

Additional model metrics:

- rolling AP: `0.2459`;
- rolling AUC: `0.8220`;
- mean alert reduction: `97.01%`;
- mean previously alerted locations dropped: `47.55`.

The model precision is approximately 4.4 times the daily naive precision:

```text
0.1747 / 0.0396 ≈ 4.4
```

The model therefore does not merely suppress alerts. The remaining alerts are
also more concentrated on locations that subsequently burn.

### 9.2 Exact final Day-8 comparison

| Metric | Rolling model on Day 8 | Fixed Day-5 naive 4 km baseline |
|---|---:|---:|
| Mean alert count | 88.58 | 2615.44 |
| Mean precision | 0.1641 | 0.0459 |
| Alert reduction | 96.41% | Reference |

This is the clearest result for the revised temporal objective. Starting from
the same initial decision context, the rolling discriminative framework reaches
Day 8 with far fewer active alerts and substantially higher precision than the
fixed 4 km rule.

---

## 10. How to Interpret the Main Metrics

### Average Precision

AP measures precision across different score thresholds and is appropriate for
rare positive locations. The 2020 rolling AP of `0.2459` is far above the raw
positive rate expected from an uninformative ranking.

### Within-event or rolling AUC

An AUC of `0.8220` means that, within the evaluated candidate groups, a randomly
selected burned location will usually receive a higher risk score than a
randomly selected non-burned location.

### Precision

Precision answers:

> Of the locations alerted, what fraction actually burned on the target day?

The 2020 rolling model precision of `0.1747` is not perfect, but it is much
higher than the naive value of `0.0396`.

### Alert reduction

Alert reduction measures the decrease in alert count relative to the naive
distance rule:

```text
1 - model alerts / naive alerts
```

The `97.01%` rolling reduction and `96.41%` final-day reduction directly support
the alert-fatigue claim.

---

## 11. What the Final Results Justify

The final experiment supports four claims.

First, the model provides **location discrimination**. It ranks nearby locations
using more than a simple distance rule, and its AP exceeds the distance-only
baseline in all three 2019 held-out folds.

Second, the framework provides **rolling alert revision**. Alerts can be added,
retained or removed as new daily conditions become available.

Third, the framework provides a large **alert-volume reduction**. The external
2020 test shows a reduction above 96% against both daily and fixed naive 4 km
comparisons.

Fourth, the result has **temporal generalization evidence**. Models and
thresholds developed on 2019 retain useful discrimination on the unseen 2020
season.

---

## 12. What the Final Results Do Not Justify

The thesis should not claim:

- perfect wildfire spread prediction;
- causal explanations from SHAP;
- online model retraining after Day 6 or Day 7;
- proof of generalization to every geography or future climate condition;
- superiority to distance on every individual metric;
- operational deployment readiness without further calibration and
  decision-maker testing.

The 2020 result is substantially stronger than 2019-only cross-validation, but
it remains one external year.

---

## 13. Final Methodological Flow

The authoritative thesis flow can be summarized as:

```text
2019 fire events
    |
    v
Three mutually exclusive event folds
    |
    v
Train three Stage 1 U-Net models
    |
    v
Generate leakage-free held-out Stage 1 predictions
    |
    v
Create fair 4 km candidate locations
    |
    v
Train three Stage 2 LightGBM models
    |
    v
Calibrate the alert threshold using 2019 OOF scores only
    |
    +---------------- freeze complete methodology ----------------+
                                                               |
                                                               v
                                                     Unseen 2020 events
                                                               |
                                                               v
                                             Three Stage 1 predictions
                                                               |
                                                               v
                                              Average probability rasters
                                                               |
                                                               v
                                              Fair candidate engineering
                                                               |
                                                               v
                                             Three Stage 2 model scores
                                                               |
                                                               v
                                                Average alert probabilities
                                                               |
                                                               v
                                    Rolling 2 km -> 3 km -> 4 km alerts
                                                               |
                                                               v
                         Compare with daily and fixed Day-5 naive 4 km alerts
```

The central methodological improvement is therefore not merely adding another
dataset year. It is the combination of:

- event-independent validation;
- rolling temporal decisions;
- frozen out-of-fold calibration;
- an unseen-year test;
- and explicit measurement of alert fatigue.

Together, these changes convert the initial prototype into a substantially more
defensible Master's thesis experiment.
