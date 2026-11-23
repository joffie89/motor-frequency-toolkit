# Efficiency benchmark

678,013 policies. Exposure-weighted one-way frequency, best of 3 runs.

| feature   |   loop_s |   vectorised_s |   vectorised_categorical_s |   speed_up_vs_loop | identical_results   |
|:----------|---------:|---------------:|---------------------------:|-------------------:|:--------------------|
| VehBrand  |   0.1112 |         0.0368 |                     0.0106 |            10.5378 | True                |
| Region    |   0.1076 |         0.0368 |                     0.011  |             9.8278 | True                |
| Area      |   0.1089 |         0.0352 |                     0.0098 |            11.0607 | True                |
| VehGas    |   0.1139 |         0.0404 |                     0.0101 |            11.2593 | True                |

Total: row loop 0.442s, vectorised 0.149s, vectorised on categoricals 0.041s (11x faster than the loop). All three give identical results.

## Memory of text columns

Strings: 142.9 MB; categorical: 2.7 MB (53x smaller).