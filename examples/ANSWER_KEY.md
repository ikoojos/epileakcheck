# Answer key for the EpiLeakCheck samples

`leaky_pipeline.py` has six planted failures. `clean_pipeline.py` uses the same data and
models with every failure fixed, so it should give no findings. Both scripts run end to end
(numpy, pandas, scikit-learn) and print their test MAE and outbreak F1.

## Planted failures in leaky_pipeline.py

| # | Line | Rule | Expected confidence | What is wrong | Fixed in clean_pipeline.py by |
| --- | --- | --- | --- | --- | --- |
| 1 | 33 | CL-11 Feature from the same-period target | high | `cases_roll4 = cases.rolling(4).mean()` includes the week being predicted | `cases.shift(1).rolling(4).mean()` |
| 2 | 36 | CL-05 Covariates aligned to future weeks | medium | `temp_avg`, `rainfall`, `humidity` of the target week are inputs | lagged copies `*_lag1` |
| 3 | 42 | CL-07 Preprocessing statistics from evaluation data | medium | `StandardScaler` fitted on all rows before the split | scaler fitted on `train_df` only |
| 4 | 45 | FM1 Temporal leakage via random split | high | `train_test_split` shuffles 10 years of weekly rows | chronological train / validation / test slices |
| 5 | 58 | CL-09 Best model chosen on the test set | medium | the model with the lowest test MAE is picked and its test MAE reported | model chosen on validation MAE, test scored once |
| 6 | 63 | CL-10 Outbreak threshold from evaluation data | medium | the 75th-percentile threshold uses all years, test included | threshold from training years |

## Result with EpiLeakCheck 0.3.1

```
epileakcheck scan leaky_pipeline.py     -> 6 findings: lines 33, 36, 42, 45, 58, 63 (all six planted, no extra)
epileakcheck scan clean_pipeline.py     -> 0 findings (at every confidence level)
```

Version 0.3.0 found five of the six; it missed #6 because it only flagged full-series
thresholds computed *before* the split line. Building these samples exposed that, and 0.3.1
fixes it (a threshold from full-role data includes the test period wherever it is computed).
The fix does not change the corpus results.

## Effect of the failures when the scripts run

On this synthetic data, the leaky script reports a test MAE of 11.88 and the clean script
13.93. The leaky number looks better because of the planted failures, not because the model
is better. That gap illustrates the optimism these failures create.

## Ideas for your own tests

- Fix one failure at a time in `leaky_pipeline.py` and check that only that finding disappears.
- Rename `X_test` to something neutral (e.g. `X2`) to see a known limitation: roles come from names.
- Move the scaler into another file to see the single-file data-flow limitation.
