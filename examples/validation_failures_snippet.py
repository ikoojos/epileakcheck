import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score

sinan = pd.read_csv("DENGBR24.csv", usecols=["DT_NOTIFIC", "DT_SIN_PRI", "ID_MUNICIP"])
sinan["DT_NOTIFIC"] = pd.to_datetime(sinan["DT_NOTIFIC"])
df = sinan.groupby(sinan["DT_NOTIFIC"].dt.to_period("W")).size().to_frame("cases")

df["cases_lag1"] = df["cases"].shift(1)
df["cases_ewm"] = df["cases"].ewm(span=4).mean()
df["target_4w"] = df["cases"].rolling(4).sum().shift(1 - 4)
df = df.dropna()

X = df[["cases_lag1", "cases_ewm"]]
y = df["target_4w"]

model = GradientBoostingRegressor()
scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True))

def outbreak_f1(y_true, y_pred):
    thr = np.quantile(y_true, 0.9)
    return f1_score(y_true > thr, y_pred > thr)
