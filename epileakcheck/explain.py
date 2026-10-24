"""Explanations and remediation for every sub-pattern a detector can report.

Each entry: what (the problem in one sentence), why (why it inflates or distorts the reported
performance), how (ordered fix steps), before / after (minimal code illustrating the fix).
`related` names the kind of supporting line the scanner looks up for this pattern.
"""
from __future__ import annotations

E: dict[str, dict] = {}


def _e(pattern, what, why, how, before, after, related=None):
    E[pattern] = {"what": what, "why": why, "how": how, "before": before.strip("\n"),
                  "after": after.strip("\n"), "related": related or []}


# ------------------------------------------------------------------ FM1
_e("random-train-test-split",
   "Time-ordered rows are split into train and test at random.",
   "Test weeks are surrounded by training weeks from before and after them, so the model "
   "interpolates instead of forecasting. Reported errors are lower than what a real forecast achieves.",
   ["Sort the data by date.",
    "Split by position or date: train_test_split(..., shuffle=False) or a date cut-off.",
    "Prefer rolling-origin evaluation (TimeSeriesSplit or an expanding window) for the final score."],
   """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""", """
df = df.sort_values("date")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# or: tscv = TimeSeriesSplit(n_splits=5)
""", ["time"])
_e("non-temporal-cv",
   "K-fold style cross-validation is used on time-indexed data.",
   "Folds are not ordered in time, so most folds are predicted by a model trained on later periods. "
   "This holds even with shuffle=False.",
   ["Replace the splitter with TimeSeriesSplit (or a walk-forward loop).",
    "If the folds feed a stacking meta-learner, build its out-of-fold predictions with TimeSeriesSplit too."],
   """
kf = KFold(n_splits=5, shuffle=False)
""", """
kf = TimeSeriesSplit(n_splits=5)
""", ["time"])
E["default-kfold"] = dict(E["non-temporal-cv"], what="A cross-validation helper is called without cv=, so it uses "
                          "(Stratified)KFold, which ignores time order.",
                          before="scores = cross_val_score(model, X, y)",
                          after="scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))")
E["integer-cv"] = dict(E["non-temporal-cv"], what="cv=<integer> requests k-fold splits that ignore time order.",
                       before="search = GridSearchCV(model, grid, cv=5)",
                       after="search = GridSearchCV(model, grid, cv=TimeSeriesSplit(n_splits=5))")
_e("reversed-before-positional-split",
   "The series is put in reverse time order before it is split by position.",
   "The first part, used for training, is the most recent data, and the test set lies in the past. "
   "The model is evaluated on hindcasting, not forecasting.",
   ["Keep the data in ascending date order (sort_values('date')).",
    "Remove the reversal ([::-1]) before splitting, or reverse back afterwards and split again."],
   """
TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocessor(df[::-1], window)
""", """
df = df.sort_values("date")            # oldest first
TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocessor(df, window)
""", ["time"])

# ------------------------------------------------------------------ FM2
_e("notification-anchor-no-correction",
   "Cases are counted by notification date, and the latest periods are used as if complete.",
   "Notifications for recent weeks keep arriving for weeks or months. At forecast time the last "
   "weeks are undercounted, but the training and test data use the final counts, so the evaluation "
   "assumes information the forecaster will not have.",
   ["Use data vintages: rebuild each origin's training data as it was known then (by DT_DIGITA or "
    "download date).",
    "If vintages are unavailable, drop or down-weight the last k weeks (k = a high percentile of the "
    "reporting delay), or nowcast them before forecasting.",
    "Report the delay assumption with the results."],
   """
weekly = df.groupby(df["DT_NOTIFIC"].dt.to_period("W")).size()
""", """
asof = origin                                    # forecast date
known = df[df["DT_DIGITA"] <= asof]              # what was in the system at that date
weekly = known.groupby(known["DT_NOTIFIC"].dt.to_period("W")).size()
weekly = weekly[weekly.index <= asof - delay_p75]  # or nowcast the tail
""")
E["notification-anchor-onset-available"] = dict(
    E["notification-anchor-no-correction"],
    what="Cases are dated by notification date although an onset date is available, with no lag correction.",
    how=["Decide which date the target should follow (onset is closer to transmission; notification is what "
         "arrives first).",
         "Whichever you use, rebuild training data per origin from what was known then, or nowcast the "
         "incomplete recent weeks.",
         "Keep the onset date as a check: compare onset- and notification-dated series for the last weeks."])
_e("latest-vintage-revisable-source",
   "Data are downloaded as the latest snapshot of a source that revises past counts, and used for back-tests.",
   "Every past forecast date is evaluated on values that were corrected later. Inputs at each origin "
   "look cleaner and more complete than they were, so back-test errors are optimistic.",
   ["Use the source's versioned data (e.g. Delphi epidata issues=/as_of=, Forecast Hub truth by date, "
    "archived snapshots in the repository's history).",
    "For each origin, train only on the vintage available at that date.",
    "If no vintages exist, state the limitation and evaluate only recent origins with a delay buffer."],
   """
df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/.../time_series.csv")
""", """
for origin in origins:
    df = load_snapshot(as_of=origin)      # the data as published on that date
    model.fit(features(df, origin), target(df, origin))
""")
_e("provisional-file",
   "A file marked provisional or partial-period is used as if it held final totals.",
   "Its last period covers only part of the time or will be revised upward, so it looks like a fall in "
   "cases and distorts both training and evaluation.",
   ["Exclude the provisional period from training and testing, or",
    "scale it to a full period with an explicit, documented completion factor, and",
    "re-run once the final figures are published."],
   """
df = pd.read_csv("cases_2021_2026_till_April.csv")
""", """
df = pd.read_csv("cases_2021_2026_till_April.csv")
df = df[df["year"] < 2026]               # 2026 is provisional (Jan-Apr only)
""")

# ------------------------------------------------------------------ FM3
_e("spatial-feature-from-all-units",
   "Units are held out for testing, but a spatial feature is computed from all units, including the "
   "held-out ones.",
   "The held-out units' own values reach the training data through neighbour averages or spatial lags, "
   "so the model has partly seen the units it is tested on.",
   ["Split units first.",
    "Compute neighbour/spatial features for training rows from training units only.",
    "For test units, compute them from information that would be available for an unseen unit."],
   """
df["neighbor_mean"] = df.groupby("week")["cases"].transform("mean")
for tr, te in GroupKFold(5).split(X, y, groups=df["municipality"]): ...
""", """
for tr, te in GroupKFold(5).split(X, y, groups=df["municipality"]):
    train = df.iloc[tr].copy()
    train["neighbor_mean"] = train.groupby("week")["cases"].transform("mean")
    # build test features without the test units' own counts
""", ["split"])
_e("random-split-transfer-claim",
   "Rows of a multi-unit panel are split at random while the code or documentation claims generalisation "
   "to other units.",
   "The same units (and neighbouring weeks of them) appear in train and test, so the score says nothing "
   "about performance on new units.",
   ["Hold out whole units with GroupKFold / LeaveOneGroupOut, or spatial blocks.",
    "Or drop the transfer claim and evaluate temporally."],
   """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
""", """
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(X, y, groups=df["municipality"]): ...
""")

# ------------------------------------------------------------------ FM4
_e("multi-year-load-unharmonised",
   "Files for several years are loaded and combined without harmonising columns or codes.",
   "Surveillance schemas change over time (e.g. SINAN's 2013/2014 change from one- to two-digit codes, "
   "renamed fields, split states). Unharmonised values mean different things in different years, so "
   "trends and test periods mix incompatible definitions.",
   ["List each year's columns and codes; keep the intersection or map them to one schema.",
    "Apply year-conditional recodes for known classification changes.",
    "Check yearly totals for breaks (sudden drops or jumps) before modelling."],
   """
frames = [pd.read_csv(f"DENGBR{y}.csv") for y in range(2010, 2025)]
df = pd.concat(frames)
""", """
frames = []
for y in range(2010, 2025):
    d = pd.read_csv(f"DENGBR{y}.csv")
    d = d.rename(columns=COLUMN_MAP.get(y, {}))[COMMON_COLS]
    d["CLASSI_FIN"] = recode_classification(d["CLASSI_FIN"], year=y)
    frames.append(d)
df = pd.concat(frames)
""")
E["concat-year-files"] = dict(E["multi-year-load-unharmonised"],
                              what="Year-specific files are concatenated without harmonising columns or codes.")

# ------------------------------------------------------------------ CL-05
_e("negative-shift-covariate",
   "A covariate is shifted backwards, so each row holds a future value of it.",
   "The model uses information from after the forecast date, which is unavailable in deployment.",
   ["Use shift(k) with k >= horizon for covariates (positive lags only).",
    "If future covariates are intended, use forecasts available at the origin, not observed values."],
   """
df["rain_next"] = df["rain"].shift(-2)
""", """
df["rain_lag2"] = df["rain"].shift(2)
""", ["target"])
_e("same-period-covariates",
   "Covariates from the same period as the target are model inputs.",
   "Same-week weather or surveillance values are not known when a forecast for that week is made. "
   "The model looks better than any real forecast.",
   ["Lag every covariate by at least the forecast horizon (shift(h)), or",
    "shift the target forward (target = cases.shift(-h)) so all inputs precede it, or",
    "use covariate forecasts available at the origin."],
   """
features = ["temp_avg", "rainfall", "cases_lag1"]
y = df["cases"]
""", """
h = 4
for c in ["temp_avg", "rainfall"]:
    df[f"{c}_lag{h}"] = df[c].shift(h)
features = [f"temp_avg_lag{h}", f"rainfall_lag{h}", "cases_lag1"]
""", ["target"])
E["same-period-covariates-from-data"] = dict(
    E["same-period-covariates"],
    what="All data columns except the target are used as inputs, and the data contain same-period covariates.",
    before='X = df.drop(columns=["cases"])',
    after='X = df.drop(columns=["cases"]).shift(h)   # or select lagged columns explicitly')
_e("lag-search-includes-zero",
   "A covariate lag search includes lag 0.",
   "Lag 0 means the covariate of the target week itself, which is unknown at forecast time. If it is "
   "selected, the model uses same-period information.",
   ["Start the search at the forecast horizon: range(h, max_lag + 1).",
    "Run the search on training data only."],
   """
for lag in range(max_lag + 1):
""", """
for lag in range(horizon, max_lag + 1):
""")

# ------------------------------------------------------------------ CL-06
_e("rolling-target-starts-at-t",
   "The target is a rolling sum shifted by 1 - window, so it starts at the last row used as input.",
   "The first period of the 'h-ahead' target is already observed in the inputs, and the horizon is "
   "counted from the data period, not from when the forecast can be issued.",
   ["Start the target window after the last input period: shift(-window).",
    "Count the horizon from forecast availability (add the publication lag).",
    "State the lead time users actually get."],
   """
df["target"] = df["cases"].rolling(7).sum().shift(1 - 7)
""", """
df["target"] = df["cases"].rolling(7).sum().shift(-7)   # days t+1 .. t+7
""")

# ------------------------------------------------------------------ CL-07
_e("fit-on-test",
   "A scaler, imputer or encoder is fitted on test data.",
   "The test set's statistics shape the inputs the model is evaluated on, and the test rows are scaled "
   "differently from training rows.",
   ["Fit on training data only: scaler.fit(X_train).",
    "Apply to test with transform(X_test), never fit_transform."],
   """
X_test_s = scaler.fit_transform(X_test)
""", """
scaler.fit(X_train)
X_test_s = scaler.transform(X_test)
""")
E["fit-on-validation"] = dict(E["fit-on-test"], what="A scaler, imputer or encoder is fitted on validation data.",
                              before="X_val_s = scaler.fit_transform(X_val)",
                              after="scaler.fit(X_train)\nX_val_s = scaler.transform(X_val)")
_e("fit-before-split",
   "A scaler, imputer or encoder is fitted on the full data before the train/test split.",
   "Its statistics (mean, min, max, imputed values) include the evaluation period, so information "
   "from the test period reaches training inputs.",
   ["Split first.",
    "Fit on the training part only, then transform validation and test.",
    "In walk-forward loops, refit at every origin."],
   """
data = scaler.fit_transform(df)
train, test = data[:n], data[n:]
""", """
train, test = df[:n], df[n:]
scaler.fit(train)
train_s, test_s = scaler.transform(train), scaler.transform(test)
""", ["split"])
_e("eval-normalised-with-own-stats",
   "Evaluation data are normalised with their own mean/std (or min/max).",
   "The test inputs are scaled using statistics of the test period itself, including its future rows, "
   "and on a different scale from training.",
   ["Compute normalisation statistics on training data.",
    "Reuse them for validation and test."],
   """
z_test = (data_test - data_test.mean(axis=0)) / data_test.std(axis=0)
""", """
mu, sd = data_train.mean(axis=0), data_train.std(axis=0)
z_test = (data_test - mu) / sd
""")
_e("normalisation-stat-full",
   "A normalisation statistic is computed over the whole array, not just training rows.",
   "Min/max or mean/std include the evaluation period, so test values are guaranteed to fall in the "
   "training range and extrapolation is hidden.",
   ["Compute the statistic on training rows only.",
    "Apply the same statistic to the rest."],
   """
max_, min_ = np.max(data, axis=1), np.min(data, axis=1)
norm = (data - min_) / (max_ - min_)
""", """
tr = data[:, :n_train]
max_, min_ = tr.max(axis=1, keepdims=True), tr.min(axis=1, keepdims=True)
norm = (data - min_) / (max_ - min_)
""", ["split"])
_e("fillna-full-stat",
   "Missing values are filled with a statistic of the full series before the split.",
   "The fill values depend on the evaluation period.",
   ["Split first, compute the fill statistic on training rows, then fill train and test with it.",
    "Or use SimpleImputer fitted on training data."],
   """
df["temp"] = df["temp"].fillna(df["temp"].mean())
""", """
fill = train["temp"].mean()
train["temp"] = train["temp"].fillna(fill)
test["temp"] = test["temp"].fillna(fill)
""", ["split"])
_e("impute-full-array-stats",
   "Missing values are filled with column statistics of the whole array.",
   "The imputed values depend on the evaluation period.",
   ["Compute column statistics on the training rows.",
    "Fill training and test rows with those values."],
   """
col_means = np.nanmean(exog, axis=0)
exog[r, c] = col_means[c]
""", """
col_means = np.nanmean(exog[:train_end], axis=0)
exog[r, c] = col_means[c]
""", ["split"])
_e("backward-fill",
   "Backward fill copies later values into earlier rows.",
   "Before the split it moves evaluation-period values into training rows. Within a series it always "
   "uses the future.",
   ["Use forward fill (past values) or a training-period statistic.",
    "If leading gaps need values, fill them from training data only."],
   """
df = df.ffill().bfill()
""", """
df = df.ffill()            # past values only; leave or drop leading gaps
""", ["split"])
_e("interpolate-both",
   "interpolate(limit_direction='both') fills gaps from later values over the whole series.",
   "Interpolated training values use observations from the evaluation period.",
   ["Interpolate after splitting, within each part.",
    "Use limit_direction='forward' so values come from the past."],
   """
df["prcp"] = df["prcp"].interpolate(limit_direction="both")
""", """
train["prcp"] = train["prcp"].interpolate(limit_direction="forward")
""", ["split"])

# ------------------------------------------------------------------ CL-09
_e("early-stop-on-test",
   "The test split is used for early stopping or model monitoring.",
   "Training stops at the epoch that does best on the test set, and that same test score is reported. "
   "The test set has become a validation set, and the reported score is optimistic.",
   ["Carve a validation split from the end of the training period.",
    "Monitor that for early stopping; evaluate on test once."],
   """
model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[es])
""", """
X_tr, X_val = X_train[:-k], X_train[-k:]
y_tr, y_val = y_train[:-k], y_train[-k:]
model.fit(X_tr, y_tr, validation_data=(X_val, y_val), callbacks=[es])
print(evaluate(model, X_test, y_test))   # once
""")
_e("rank-test-scores",
   "Models are ranked by their test-set scores and the winner is reported with that score.",
   "Choosing the best of several models on the test set and reporting its test score overstates "
   "performance (selection bias).",
   ["Rank models on a validation split or inner time-series CV.",
    "Report the chosen model's score on the untouched test set."],
   """
scores[name] = mean_absolute_error(y_test, m.predict(X_test))
best = min(scores, key=scores.get)
""", """
val_scores[name] = mean_absolute_error(y_val, m.predict(X_val))
best = min(val_scores, key=val_scores.get)
test_mae = mean_absolute_error(y_test, models[best].predict(X_test))
""", ["metric"])
E["select-by-test-metric"] = dict(E["rank-test-scores"],
                                  what="The best model is picked by an error metric, and the only scores computed "
                                       "are on the test or hold-out split.")
E["keep-best-on-test"] = dict(E["rank-test-scores"],
                              what="A best-so-far loop keeps the configuration, epoch or feature set with the "
                                   "lowest test error.",
                              before="if rmse_test < best_rmse: best = cfg",
                              after="if rmse_val < best_rmse: best = cfg   # validation, not test")
_e("order-search-full-series",
   "The model order (e.g. ARIMA p,d,q) is chosen on the full series, test period included.",
   "Information criteria and stationarity tests computed over the full series reflect the test period, "
   "so the chosen order is tuned to it.",
   ["Select the order on the training part only (auto_arima(train) or AIC on train).",
    "Refit at each origin in walk-forward evaluation."],
   """
order = auto_arima(series).order
""", """
order = auto_arima(series[:train_end]).order
""", ["split"])
_e("search-on-test",
   "A hyper-parameter search is fitted on test data.",
   "Hyper-parameters are tuned to the test set.",
   ["Fit the search on training data with a time-series CV.",
    "Evaluate the best estimator once on test."],
   """
search.fit(X_test, y_test)
""", """
search = GridSearchCV(model, grid, cv=TimeSeriesSplit(5)).fit(X_train, y_train)
""")

# ------------------------------------------------------------------ CL-10
_e("threshold-from-full-series",
   "An outbreak or alert threshold is computed from the whole series, evaluation period included.",
   "Event labels in the test period depend on the test period's own distribution. F1 and recall are "
   "computed against a threshold no forecaster could have set in advance.",
   ["Compute the threshold from training years only (or use an external, fixed rule).",
    "Keep it fixed when scoring the test period."],
   """
threshold = df["cases"].quantile(0.75)
""", """
threshold = train["cases"].quantile(0.75)
""", ["split"])
E["threshold-from-test"] = dict(E["threshold-from-full-series"],
                                what="An outbreak or alert threshold is computed from the test data.",
                                before="thr = y_test.quantile(0.9)", after="thr = y_train.quantile(0.9)")
E["threshold-from-scored-labels"] = dict(
    E["threshold-from-full-series"],
    what="The threshold is computed inside the scoring function from the labels being scored.",
    before="def outbreak_f1(y_true, y_pred):\n    thr = np.quantile(y_true, 0.9)",
    after="def outbreak_f1(y_true, y_pred, thr):    # thr fixed from training data\n    ...")

# ------------------------------------------------------------------ CL-11
_e("unshifted-rolling-on-target",
   "A rolling/cumulative/difference feature is computed on the target without a shift.",
   "Each row's feature includes that row's own target value, so the label leaks into the inputs. "
   "With lag features as well, the target can be reconstructed exactly.",
   ["Shift the target before rolling: df[y].shift(1).rolling(k).mean().",
    "Or shift the result: df[y].rolling(k).mean().shift(1).",
    "For grouped panels, shift inside the group: groupby(unit)[y].transform(lambda s: s.shift(1).rolling(k).mean())."],
   """
df["rolling_mean_3"] = df["cases"].rolling(3).mean()
""", """
df["rolling_mean_3"] = df["cases"].shift(1).rolling(3).mean()
""", ["target", "feature"])
_e("target-arithmetic",
   "A feature is an arithmetic function of the same-period target (ratio, z-score, per-capita rate).",
   "The label enters the inputs directly. The model can recover the target from the feature.",
   ["Build such features from lagged target values only: df[y].shift(1) / df[pop].",
    "Or remove them from the input list."],
   """
df["incidence_per_1k"] = df["cases"] / df["pop"] * 1000
""", """
df["incidence_per_1k_lag1"] = df["cases"].shift(1) / df["pop"] * 1000
""", ["target", "feature"])


def explain(pattern: str) -> dict | None:
    return E.get(pattern)
