"""Fixtures for FM1-FM4 detectors."""
import textwrap

from epileakcheck import scan_source


def rules(src: str, min_conf: str = "low") -> set[str]:
    return scan_source(textwrap.dedent(src), "t.py", min_confidence=min_conf).rules_hit()


def test_fm1_default_shuffle():
    assert "FM1" in rules("""
        df['date'] = pd.to_datetime(df['date'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """, "high")

def test_fm1_shuffle_false_is_fine():
    assert "FM1" not in rules("""
        df['date'] = pd.to_datetime(df['date'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    """)

def test_fm1_kfold_and_cv_int():
    assert "FM1" in rules("""
        df['lag1'] = df['cases'].shift(1)
        scores = cross_val_score(model, X, y, cv=5)
    """)
    assert "FM1" in rules("""
        df['week'] = df.index.week
        kf = KFold(n_splits=5, shuffle=True)
    """, "high")

def test_fm1_timeseriessplit_is_fine():
    assert "FM1" not in rules("""
        df['week'] = df.index.week
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X, y, cv=tscv)
    """)

def test_fm1_reversed_series():
    assert "FM1" in rules("""
        df = pd.read_csv('d.csv', parse_dates=['date'])
        df = df[::-1]
        train, test = df[:300], df[300:]
    """)

def test_fm2_notification_anchor():
    assert "FM2" in rules("""
        df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'])
        weekly = df.groupby(df['DT_NOTIFIC'].dt.isocalendar().week).size()
        X_train, X_test = weekly[:-8], weekly[-8:]
    """)

def test_fm2_onset_available_is_high():
    assert "FM2" in rules("""
        df = pd.read_csv('sinan.csv', usecols=['DT_NOTIFIC', 'DT_SIN_PRI'])
        df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'])
        weekly = df.groupby(df['DT_NOTIFIC'].dt.to_period('W')).size()
    """, "high")

def test_fm2_vintage_guard_suppresses():
    assert "FM2" not in rules("""
        df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'])
        data = load_vintage(as_of=origin)
    """)

def test_fm2_latest_snapshot_download():
    assert "FM2" in rules("""
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/x.csv'
        df = pd.read_csv(url)
        print(mean_absolute_error(y_test, pred))
    """)

def test_fm3_spatial_feature_with_unit_holdout():
    assert "FM3" in rules("""
        df['neighbor_mean'] = df.groupby('week')['cases'].transform('mean')
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(X, y, groups=df['municipality']):
            pass
    """)

def test_fm3_temporal_panel_is_fine():
    assert "FM3" not in rules("""
        df['neighbor_mean'] = df.groupby('week')['cases'].transform('mean')
        train = df[df['year'] < 2020]; test = df[df['year'] >= 2020]
        g = df.groupby('municipality')
    """)

def test_fm4_year_loop():
    assert "FM4" in rules("""
        frames = []
        for year in range(2010, 2025):
            frames.append(pd.read_csv(f'DENGBR{year}.csv'))
        df = pd.concat(frames)
    """)

def test_fm4_harmonised_is_fine():
    assert "FM4" not in rules("""
        frames = []
        for year in range(2010, 2025):
            d = pd.read_csv(f'DENGBR{year}.csv')
            frames.append(d.rename(columns=COLUMN_MAP[year]))
        df = pd.concat(frames)
    """)
