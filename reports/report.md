# Model and feature report

_Public French motor data (freMTPL2). Periods and drift are simulated: illustrative only._

- Policies: 678,013
- Overall frequency: 0.1042 claims per vehicle-year

## Champion vs challenger (tested on a later period)

| Model | Poisson deviance | O/E | Lift (top/bottom decile) |
| --- | --- | --- | --- |
| GLM | 0.6424 | 1.133 | 7.96x |
| GBM | 0.6281 | 1.122 | 9.55x |

![Lift chart](lift_chart.png)

## Why time-based validation

Segment VehBrand=B12 has a simulated risk increase in the latest period.

- Time-based split: segment O/E = **1.56** (the shift is visible)
- Random split: segment O/E = **1.05** (the shift is hidden)

## Location grouping

| Location feature | Poisson deviance | Lift |
| --- | --- | --- |
| grouped regions (5 bands) | 0.64237 | 7.96x |
| raw Region (22 levels) | 0.64205 | 7.91x |
| without region (Area and Density only) | 0.64234 | 7.79x |

## Feature selection

| feature    |   deviance_increase_when_dropped |   one_way_spread |   stability_ratio | recommendation             |
|:-----------|---------------------------------:|-----------------:|------------------:|:---------------------------|
| VehAge     |                           0.0115 |           2.6784 |            1.1878 | keep                       |
| BonusMalus |                           0.0103 |           2.6542 |            1.0806 | keep                       |
| DrivAge    |                           0.0019 |           1.5971 |            1.0753 | keep                       |
| VehBrand   |                           0.0018 |           2.1383 |            1.4381 | review: unstable over time |
| VehPower   |                           0.0009 |           1.3075 |            1.1448 | keep                       |
| VehGas     |                           0.0001 |           1.0635 |            1.0037 | keep                       |
| Area       |                          -0      |           1.7665 |            1.1112 | candidate to drop or merge |
| Density    |                          -0      |           1.6883 |            1.0675 | candidate to drop or merge |
| AreaGroup  |                          -0      |           1.4001 |            1.0965 | candidate to drop or merge |

## Monitoring (segment O/E, latest period)

| VehBrand   |   exposure |   observed |   expected |    oe | alert   |
|:-----------|-----------:|-----------:|-----------:|------:|:--------|
| B12        |   16157.7  |       3472 |   2218.59  | 1.565 | True    |
| B3         |    7024.1  |        743 |    676.841 | 1.098 | False   |
| B11        |    1686.58 |        180 |    171.768 | 1.048 | False   |
| B2         |   23736.2  |       2176 |   2122.45  | 1.025 | False   |
| B5         |    4996.93 |        506 |    501.154 | 1.01  | False   |
| B4         |    3532.3  |        332 |    336.613 | 0.986 | False   |
| B10        |    2391.74 |        210 |    215.876 | 0.973 | False   |
| B1         |   23707    |       2061 |   2182.38  | 0.944 | False   |
| B6         |    3914.17 |        341 |    375.84  | 0.907 | False   |
| B13        |    1659.75 |        139 |    163.455 | 0.85  | False   |

Alerts: B12

## Decision

Recommendation: **promote challenger**. Status: **awaiting_approval** (promotion requires a named reviewer: `python -m motorfreq approve --by "Name"`).