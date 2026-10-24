"""Weekly dengue forecasting pipeline (synthetic data).

A deliberately flawed example for testing EpiLeakCheck. It runs end to end with
numpy, pandas and scikit-learn and prints test scores. See ANSWER_KEY.md for the
failures planted in it.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
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

# ---------------------------------------------------------------- features
df["cases_lag1"] = df["cases"].shift(1)
df["cases_lag2"] = df["cases"].shift(2)
df["cases_roll4"] = df["cases"].rolling(4).mean()
df = df.dropna().reset_index(drop=True)

features = ["temp_avg", "rainfall", "humidity", "cases_lag1", "cases_lag2", "cases_roll4"]
X = df[features]
y = df["cases"]

# ---------------------------------------------------------------- scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------- split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------- model comparison
models = {
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(n_estimators=200, random_state=0),
    "gbm": GradientBoostingRegressor(random_state=0),
}
scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    scores[name] = mean_absolute_error(y_test, model.predict(X_test))

best_name = min(scores, key=scores.get)
best_model = models[best_name]
print(f"Best model: {best_name}  test MAE = {scores[best_name]:.2f}")

# ---------------------------------------------------------------- outbreak alerts
outbreak_threshold = df["cases"].quantile(0.75)
y_pred = best_model.predict(X_test)
alert_f1 = f1_score(y_test > outbreak_threshold, y_pred > outbreak_threshold)
print(f"Outbreak alert F1 = {alert_f1:.2f}")
