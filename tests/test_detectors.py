"""Positive and negative fixtures for each detector. Run: python -m pytest -q"""
import textwrap

import pytest

from epileakcheck import scan_source


def rules(src: str, min_conf: str = "low") -> set[str]:
    return scan_source(textwrap.dedent(src), "t.py", min_confidence=min_conf).rules_hit()


# ---------------- CL-11 ----------------
def test_cl11_unshifted_rolling_on_target():
    assert "CL-11" in rules("""
        df['rm3'] = df['cases'].rolling(3).mean()
        X = df[['rm3']]; y = df['cases']
    """)

def test_cl11_shifted_rolling_is_fine():
    assert "CL-11" not in rules("""
        df['rm3'] = df['cases'].shift(1).rolling(3).mean()
        df['rm3b'] = df['cases'].rolling(3).mean().shift(1)
        y = df['cases']
    """)

def test_cl11_groupby_transform():
    assert "CL-11" in rules("""
        TARGET = 'casos'
        df['m'] = df.groupby('city')['casos'].transform(lambda s: s.rolling(4).mean())
    """)

def test_cl11_ratio_of_target():
    assert "CL-11" in rules("""
        y = df['cases']
        df['inc_per_1k'] = df['cases'] / df['pop'] * 1000
        features = ['inc_per_1k', 'temp_lag1']
    """)

def test_cl11_display_ratio_not_flagged():
    assert "CL-11" not in rules("""
        y = df['cases']
        df['inc_per_1k'] = df['cases'] / df['pop'] * 1000
        plot(df['inc_per_1k'])
    """)


# ---------------- CL-06 ----------------
def test_cl06_rolling_target_starts_at_t():
    assert "CL-06" in rules("""
        h = 7
        df['target'] = df['confirmed'].rolling(h).sum().shift(1 - h)
    """)

def test_cl06_negative():
    assert "CL-06" not in rules("""
        df['target'] = df['confirmed'].rolling(7).sum().shift(-7)
    """)


# ---------------- CL-05 ----------------
def test_cl05_negative_shift_covariate():
    assert "CL-05" in rules("""
        y = df['cases']
        df['rain_next'] = df['rain'].shift(-2)
    """)

def test_cl05_same_period_weather():
    assert "CL-05" in rules("""
        features = ['temp_avg', 'rainfall', 'cases_lag1']
        X = df[features]; y = df['cases']
    """)

def test_cl05_lagged_weather_or_future_target_is_fine():
    assert "CL-05" not in rules("""
        features = ['temp_lag2', 'rainfall_lag4', 'cases_lag1']
        y = df['cases']
    """)
    assert "CL-05" not in rules("""
        df['target'] = df['cases'].shift(-4)
        features = ['temp_avg', 'rainfall']
        y = df['target']
    """)


# ---------------- CL-07 ----------------
def test_cl07_fit_before_split():
    assert "CL-07" in rules("""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df[['cases']])
        train, test = data[:100], data[100:]
    """)

def test_cl07_fit_on_test():
    assert "CL-07" in rules("""
        sc = StandardScaler()
        X_test_s = sc.fit_transform(X_test)
    """)

def test_cl07_fit_on_train_is_fine():
    assert "CL-07" not in rules("""
        train, test = df[:100], df[100:]
        sc = StandardScaler()
        Xtr = sc.fit_transform(train)
        Xte = sc.transform(test)
    """)


# ---------------- CL-09 ----------------
def test_cl09_early_stopping_on_test():
    assert "CL-09" in rules("""
        model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[es])
    """)

def test_cl09_rank_test_scores():
    assert "CL-09" in rules("""
        scores = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            scores[name] = mean_absolute_error(y_test, m.predict(X_test))
        best = min(scores, key=scores.get)
    """)

def test_cl09_validation_selection_is_fine():
    assert "CL-09" not in rules("""
        scores = {}
        for name, m in models.items():
            scores[name] = mean_absolute_error(y_val, m.predict(X_val))
        best = min(scores, key=scores.get)
        print(mean_absolute_error(y_test, models[best].predict(X_test)))
    """)


# ---------------- CL-10 ----------------
def test_cl10_threshold_full_series():
    assert "CL-10" in rules("""
        threshold = df['cases'].quantile(0.75)
        df['outbreak'] = df['cases'] > threshold
        train, test = df[:200], df[200:]
        print(f1_score(test['outbreak'], pred > threshold))
    """)

def test_cl10_threshold_from_scored_labels():
    assert "CL-10" in rules("""
        def outbreak_f1(y_true, y_pred):
            thr = np.quantile(y_true, 0.9)
            return f1_score(y_true > thr, y_pred > thr)
    """)

def test_cl10_train_threshold_is_fine():
    assert "CL-10" not in rules("""
        train, test = df[:200], df[200:]
        threshold = train['cases'].quantile(0.75)
    """)



def test_api_reports_fields():
    rep = scan_source("sc = StandardScaler()\nZ = sc.fit_transform(X_test)\n", "a.py")
    f = rep.findings[0]
    assert f.rule == "CL-07" and f.line == 2 and f.confidence == "high" and f.kn_type == "L1.2"


# ---------------- regression fixtures for v0.2 patterns ----------------
def test_cl07_test_normalised_with_own_stats():
    assert "CL-07" in rules("""
        data_test = data[n:]
        mean_test = np.mean(data_test, axis=0)
        std_test = np.std(data_test, axis=0)
        z = (data_test - mean_test) / std_test
    """, "high")

def test_cl07_full_array_imputation():
    assert "CL-07" in rules("""
        model = SARIMAX(y_train, exog=exog_train)
        col_means = np.nanmean(exog, axis=0)
        exog[rows, cols] = col_means[cols]
    """)

def test_cl05_covariate_lag_search_from_zero():
    assert "CL-05" in rules("""
        def select_weather_lags(train):
            for lag in range(13):
                c = train['rainfall'].shift(lag).corr(train['cases'])
        model.fit(X, y)
    """, "medium")

def test_cl09_keep_best_on_test_metric():
    assert "CL-09" in rules("""
        y_te = df_te['cases']
        for cols in subsets:
            row = {'rmse': rmse(y_te, model.predict(X_te[cols]))}
            if best is None or row['rmse'] < best['rmse']:
                best = row
    """)

def test_cl09_alias_to_holdout():
    assert "CL-09" in rules("""
        y_true = stack['y_ho_raw'].values
        metrics = compute_metrics(y_true, y_pred)
        best = min(city_results, key=lambda r: r.mae)
    """)


# ---------------- v0.3: segments, related lines and explanations ----------------
def test_finding_has_segment_related_and_fix():
    src = textwrap.dedent("""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df[['cases', 'rain']])
        train, test = data[:100], data[100:]
    """)
    f = [x for x in scan_source(src, "t.py").findings if x.rule == "CL-07"][0]
    assert f.line == 4 and f.end_line == 4
    assert any(s["flagged"] and "fit_transform" in s["code"] for s in f.segment)
    assert any(r["line"] == 5 and "split" in r["label"] for r in f.related)
    assert f.what and f.why and len(f.how) >= 2 and "transform" in f.example_after

def test_multiline_statement_span():
    src = "X_train, X_test, y_train, y_test = train_test_split(\n    X, y,\n    test_size=0.2)\ndf['date'] = 1\n"
    f = [x for x in scan_source(src, "t.py").findings if x.rule == "FM1"][0]
    assert (f.line, f.end_line) == (1, 3)

def test_label_encoder_before_split_not_flagged():
    assert "CL-07" not in rules("""
        le = LabelEncoder()
        y = le.fit_transform(df['risk'])
        X_train, X_test = X[:80], X[80:]
    """)

def test_cl10_full_series_threshold_after_split():
    assert "CL-10" in rules("""
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
        outbreak_threshold = df['cases'].quantile(0.75)
        print(f1_score(y_test > outbreak_threshold, pred > outbreak_threshold))
    """)
