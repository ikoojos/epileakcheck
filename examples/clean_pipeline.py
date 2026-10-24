"""Weekly dengue forecasting pipeline (synthetic data), corrected version.

Same data and models as leaky_pipeline.py, with the six planted failures fixed.
EpiLeakCheck should report nothing here. Remaining caveats (see ANSWER_KEY.md): a single
test window from one origin, and no reporting-delay handling for real surveillance data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- data
weeks = pd.date_range("2015-01-04", periods=520, freq="W")
season = 1 + np.sin(2 * np.pi * np.arange(520) / 52)
df = pd.DataFrame({
    "date": weeks,
    "temp_avg": 26 + 3 * season + rng.normal(0, 0.5, 520),
    "rainfall": 80 * season + rng.gamma(2, 10, 520),
    "humidity": 70 + 10 * season + rng.normal(0, 2, 520),
})
df["cases"] = np.round(50 * season ** 2 + 0.8 * df["rainfall"].shift(3).fillna(0)
                       + rng.poisson(5, 520))
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------- features (all strictly past)
H = 1  # forecast horizon in weeks
for c in ["temp_avg", "rainfall", "humidity"]:
    df[f"{c}_lag{H}"] = df[c].shift(H)
df["cases_lag1"] = df["cases"].shift(1)
df["cases_lag2"] = df["cases"].shift(2)
df["cases_roll4"] = df["cases"].shift(1).rolling(4).mean()
df = df.dropna().reset_index(drop=True)

features = ["temp_avg_lag1", "rainfall_lag1", "humidity_lag1", "cases_lag1", "cases_lag2", "cases_roll4"]

# ---------------------------------------------------------------- chronological split: train | val | test
n = len(df)
train_df = df.iloc[: int(0.6 * n)]
val_df = df.iloc[int(0.6 * n): int(0.8 * n)]
test_df = df.iloc[int(0.8 * n):]

# ---------------------------------------------------------------- scaling fitted on training rows only
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features])
X_val = scaler.transform(val_df[features])
X_test = scaler.transform(test_df[features])
y_train, y_val, y_test = train_df["cases"], val_df["cases"], test_df["cases"]

# ---------------------------------------------------------------- model selection on validation
models = {
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(n_estimators=200, random_state=0),
    "gbm": GradientBoostingRegressor(random_state=0),
}
val_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    val_scores[name] = mean_absolute_error(y_val, model.predict(X_val))

best_name = min(val_scores, key=val_scores.get)
best_model = models[best_name]
test_mae = mean_absolute_error(y_test, best_model.predict(X_test))
print(f"Best model (chosen on validation): {best_name}  test MAE = {test_mae:.2f}")

# ---------------------------------------------------------------- outbreak alerts, threshold from training years
outbreak_threshold = train_df["cases"].quantile(0.75)
y_pred = best_model.predict(X_test)
alert_f1 = f1_score(y_test > outbreak_threshold, y_pred > outbreak_threshold)
print(f"Outbreak alert F1 = {alert_f1:.2f}")
