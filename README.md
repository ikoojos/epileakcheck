# EpiLeakCheck (prototype 0.3.2)

Static checker for **validation-scheme failures** in epidemic-forecasting ML pipelines.
It reads Python scripts and Jupyter notebooks (nothing is executed) and reports the
four predefined failure modes of the ICSME 2026 registered report (FM1–FM4) plus six
candidate failure modes derived from the 34-pipeline derivation set.

**Status: research prototype.** Rules were written and tuned on the derivation set, so
the accuracy below is in-sample. The RQ3 evaluation must use the separate evaluation
set, labelled before the tool is run.

## Rules

| Rule | Failure mode | K&N | What the detector looks for |
| --- | --- | --- | --- |
| FM1 | Temporal leakage via random or non-temporal split | L3.1 | sklearn `train_test_split` with the default shuffle on time-indexed data; `KFold`/`ShuffleSplit`/`StratifiedKFold`; `cross_val_score`/`GridSearchCV` with an integer or default `cv`; series reversed (`df[::-1]`, descending date sort) before a positional split. `TimeSeriesSplit` and `shuffle=False` pass. |
| FM2 | Notification lag ignored | L3.1 | Cases dated by notification date (`DT_NOTIFIC`, `report_date`, …) with no delay correction (high when an onset date is present); latest-snapshot downloads of revisable sources (JHU CSSE, covidcast/Delphi, ILINet, InfoDengue, DATASUS) used for back-tests; files marked provisional/partial. Suppressed when the repository uses vintages, `as_of=`, nowcasting or delay triangles. |
| FM3 | Spatial leakage | L3.3 | Units held out (`GroupKFold`, `isin(test_units)`) while spatial features (neighbour means, spatial lags) are built from all units; random row splits on a multi-unit panel when the code or README claims transfer to unseen units. Temporal splits of pooled panels pass (codebook boundary rule). |
| FM4 | Schema heterogeneity without harmonisation | — | Year-keyed files loaded in a loop (`for year in range(2010, …)`, `DENGBR*`) and combined with no column intersection, mapping or per-year rename; explicit `concat` of year files. `usecols=` lowers the finding to low. |
| CL-05 | Covariates aligned to future weeks | L3.1 | Covariates shifted with a negative period; same-period weather/surveillance inputs with a same-period target (uses CSV headers found in the repo when the feature list is implicit); covariate lag searches that include lag 0. Downgraded when the repo builds sliding windows. |
| CL-06 | Horizon counted from data week | L3.1 | Rolling-sum targets shifted by `1 - h`, so the target window starts at the last input row. |
| CL-07 | Preprocessing statistics from evaluation data | L1.2 | Scalers/imputers fitted on test or validation data, or on the full data before the split; test data normalised with its own mean/std (one- or two-statement form); full-series `fillna(mean)`, whole-array `nanmean` imputation, backward fill, `interpolate(limit_direction='both')`; min/max normalisation over the full array. |
| CL-09 | Best model chosen on the test set | L1.1, L1.3 | Early stopping or `eval_set` on the test split; ranking containers of test scores; best-so-far loops on a test metric; `min(..., key=mae)` when only test/hold-out scores exist (follows `y_true = stack["y_ho"]` aliases); ARIMA order search on the full series. |
| CL-10 | Outbreak threshold from evaluation data | L1.2 | Quantile/percentile thresholds (incl. `sorted(x)[int(0.75*len(x))]`) computed on test data, on the labels being scored, or on the full series, in code that scores events (F1, recall, AUC). Plot-only modules are skipped. |
| CL-11 | Feature from the same-period target | L2 | `rolling/ewm/expanding/diff/pct_change` on the target with no shift; ratios or z-scores of the same-period target used as inputs. |

Removed in 0.2.0: CL-01, CL-02, CL-03, CL-04. Their failures depend on the data's
calendar or on what the authors claim, so static rules were mostly low-confidence
noise. Not implemented: CL-08, CL-12, CL-13, CL-14 (one pipeline each).

Confidence: **high** = the pattern is the failure; **medium** = it is the failure unless
something invisible to static analysis rules it out; **low** = a shape that often carries
the failure. Use `--min-confidence medium` for reporting.

## Install and use

```bash
pip install -e .            # no third-party dependencies (Python >= 3.9)

epileakcheck rules
epileakcheck scan path/to/repo --min-confidence medium      # text report
epileakcheck scan path/to/repo --format json                # machine-readable
epileakcheck scan path/to/repo --format sarif               # GitHub code scanning
epileakcheck scan train.py --rules FM1,CL-07
epileakcheck scan . --fail-on high                          # exit 1 for CI gates
```

`tests/` folders and `test_*.py` files are skipped unless `--include-tests` is given.

### Python API

```python
from epileakcheck import scan_path, scan_source

report = scan_path("path/to/repo", min_confidence="medium")
for f in report.findings:
    print(f.rule, f.file, f.line, f.confidence, f.message)

report = scan_source(open("train.py").read(), "train.py", rules=["FM1", "CL-11"])
report.to_dict()          # JSON-serialisable
```

### HTTP API

```bash
epileakcheck serve --port 8765
curl -X POST localhost:8765/scan -H 'Content-Type: application/json' \
     -d '{"source": "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\ndf[\"date\"]=1\n"}'
```

`GET /rules`, `GET /health`, `GET /explain/<pattern>`, `POST /scan` with `source` (code text)
or `path` (a path on the server). Standard library only; add authentication before exposing it.

### Google Colab

```python
!pip install -q git+https://github.com/<you>/epileakcheck      # or upload and unzip the package
!git clone -q https://github.com/miladrezanezhad/sir_simulator
!epileakcheck scan sir_simulator --min-confidence medium
```

### What each finding tells you

Every finding locates the failure and explains it:

| Field | Content |
| --- | --- |
| `file`, `line`, `end_line` | Where the flagged statement starts and ends (multi-line calls included) |
| `segment` | The flagged lines plus two lines of context, each with its line number and a `flagged` marker |
| `related` | Supporting lines that make it a failure, with a label: where the data are split, where the target is defined, where the column is used as an input, where the test score is computed, where rows are time-indexed |
| `what` | The problem in one sentence |
| `message` | What was found in this code, with the names involved |
| `why` | Why it distorts the reported performance |
| `how` | Ordered steps to fix it |
| `example_before`, `example_after` | Minimal code showing the fix |
| `rule`, `name`, `pattern`, `confidence`, `kn_type`, `fix` | Classification, sub-pattern, confidence, Kapoor & Narayanan type, one-line fix |

Text output (the default) prints all of this:

```
[1] CL-11 Feature computed from the same-period target  [high]
    src/.../ml_prediction.py, line 71
       70 |         # Create rolling averages
    >> 71 |         df["rolling_mean_3"] = df["cases"].rolling(window=3).mean()
       72 |         df["rolling_mean_7"] = df["cases"].rolling(window=7).mean()
    -> 91 | feature_cols = [col for col in df.columns if col not in ["cases", "day"]]
            <- inputs are all columns except the target, so 'rolling_mean_3' is included
    -> 93 | y = df["cases"]    <- the target (label) is defined here

    What:  A rolling/cumulative/difference feature is computed on the target without a shift.
    Why it matters: Each row's feature includes that row's own target value ...
    How to fix:
      1. Shift the target before rolling: df[y].shift(1).rolling(k).mean().
      ...
    Example, before:  df["rolling_mean_3"] = df["cases"].rolling(3).mean()
    after:            df["rolling_mean_3"] = df["cases"].shift(1).rolling(3).mean()
```

Other formats: `--brief` (one line and fix per finding), `--format markdown` (a report you can
paste into an issue or a review), `--format json`, `--format sarif` (start/end lines and related
locations, so GitHub shows them on the code). `epileakcheck explain FM1` or
`epileakcheck explain fit-before-split` prints the explanation without scanning.

## Development-set accuracy (in-sample)

Run on the 32 cloned derivation pipelines (P33 and P34 were pasted code without a
repository). Unit: pipeline × rule. A true positive is a rule that fires anywhere in a
pipeline where the manual coding has that FM coded P or that candidate logged.
Firings where the coders coded U are not scored. Ground truth: Claude Coding Log,
23 October 2026 (43 positives: 15 FM, 28 CL).

| Threshold | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| low and above | 32 | 9 | 11 | 0.78 | 0.74 | 0.76 |
| **medium and above** | **31** | **4** | **12** | **0.89** | **0.72** | **0.79** |
| high only | 5 | 0 | 38 | 1.00 | 0.12 | 0.21 |

Per rule at medium and above:

| Rule | TP | FP | FN | Notes |
| --- | --- | --- | --- | --- |
| FM1 | 4 | 0 | 0 | All four coded pipelines found (shuffled splits, KFold stacking, reversed series) |
| FM2 | 4 | 0 | 5 | Misses are data-provenance cases invisible in code (committed or undocumented downloads) |
| FM3 | 0 | 0 | 0 | No positives in the corpus; no false alarms in 32 pipelines |
| FM4 | 0 | 0 | 2 | Both coded cases are unit/coverage changes inside one file, not multi-year loading (P31 is found at low) |
| CL-05 | 4 | 2 | 3 | Both FPs (P02 `prepare_data.py`, P03) are same-week weather inputs the manual coding did not log |
| CL-06 | 1 | 0 | 1 | P05's case is a publication lag, not a code shape |
| CL-07 | 7 | 2 | 0 | Both FPs (P03, P17) are full-series `fillna` before the split that the manual coding did not log |
| CL-09 | 5 | 0 | 1 | Miss: P28 selects on a validation round that is also reported |
| CL-10 | 3 | 0 | 0 | |
| CL-11 | 3 | 0 | 0 | |

If the four FPs are confirmed in adjudication, precision at medium rises to 1.00.

Limitations: intra-file data flow (plus a few repository-level facts); data roles come
from names (`X_test`, `y_te`, `val_df`, `stack["y_ho"]`); target columns are inferred
from `y = df[...]`, `target = "..."` and similar; FM2 cannot see data vintages that
are not visible in code.

## Layout

```
epileakcheck/core.py          parsing, notebooks, target and split inference, Finding
epileakcheck/fm_detectors.py  FM1-FM4 (RR Detectors 1-4)
epileakcheck/detectors.py     candidate rules CL-05, CL-06, CL-07, CL-09, CL-10, CL-11
epileakcheck/explain.py       what / why / how / before-after for every sub-pattern
epileakcheck/api.py           scan_path / scan_source / Report, repository facts, related lines
epileakcheck/cli.py           command line (text, JSON, SARIF; CI exit codes)
epileakcheck/server.py        HTTP API
tests/                        positive and negative fixtures per rule (pytest)
```
