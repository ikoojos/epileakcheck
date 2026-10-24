"""Detectors 1-4 of the registered report: FM1-FM4.

FM1  Temporal leakage via random or non-temporal splitting
FM2  Notification lag ignorance (incl. retrospective evaluation on revised, latest-vintage data)
FM3  Spatial leakage (units held out while spatial features use all units, or random splits
     on a multi-unit panel when the code claims transfer to unseen units)
FM4  Schema heterogeneity without harmonisation (multi-year file loading without a column
     intersection / usecols / per-year rename)
"""
from __future__ import annotations

import ast
import re

from .core import (Finding, Module, call_name, const_number, dotted, expr_role, kwarg, names_in,
                   str_const)

FM_RULES = {
    "FM1": ("Temporal leakage via random or non-temporal split", "L3.1",
            "Split by time (train on the past, test on the future): train_test_split(..., shuffle=False) "
            "on date-sorted data, TimeSeriesSplit, or rolling-origin evaluation."),
    "FM2": ("Notification lag ignored", "L3.1",
            "Evaluate on data as it was known at each forecast origin (vintages / as-of snapshots), or "
            "exclude or nowcast the most recent, still-incomplete periods."),
    "FM3": ("Spatial leakage", "L3.3",
            "When evaluating on held-out units, build spatial features from training units only and use "
            "group-aware or spatial-block splits."),
    "FM4": ("Schema heterogeneity without harmonisation", "",
            "Harmonise yearly files before concatenating: intersect or map columns, fix code changes "
            "(e.g. 2013/2014 classification codes) and units per year."),
}

TIME_SIGNAL = re.compile(r"(to_datetime|epiweek|epi_week|\bweek|weekofyear|\bdate\b|\bdates\b|\bds\b|"
                         r"\byear\b|\bmonth\b|\.shift\(|\.rolling\(|_lag|lag_|timestamp|DatetimeIndex|"
                         r"date_range|\bt\b\s*=|time_step|lookback|horizon)", re.I)
UNIT_COLS = re.compile(r"^(municip\w*|mun_\w+|geocode|ibge\w*|cod_?mun\w*|district\w*|state\w*|uf|city|cities|"
                       r"region\w*|county\w*|province\w*|prefecture\w*|tehsil|moh|moh_area|location\w*|"
                       r"country\w*|admin\d?|node\w*|site\w*|area\w*|zone\w*|microrregiao|micro_\w+)$", re.I)
NOTIF_COLS = re.compile(r"(dt_notific|data_notific|notification_?date|date_?notif|notif_?date|report_?date|"
                        r"date_?reported|reporting_?date|dt_digita|data_registro)", re.I)
ONSET_COLS = re.compile(r"(dt_sin_pri|onset|symptom_?date|date_?symptom|dt_sintoma|first_symptom)", re.I)
REVISABLE_SOURCES = re.compile(r"(CSSEGISandData|covid19-forecast-hub|covidcast|delphi|epidata|ilinet|fluview|"
                               r"api\.mosqlimate|info\.?dengue|infodengue|ourworldindata|nytimes/covid-19-data|"
                               r"ftp\.datasus|datasus\.gov|healthdata\.gov|jhu_?csse|jhuCSSE)", re.I)
VINTAGE_GUARD = re.compile(r"(\bas_of\b|vintage|\bissues?\s*=|issue_date|snapshot_date|nowcast|reporting_?delay|"
                           r"delay_?triangle|backfill_?correction|lag_?correction|DT_DIGITA|data_?release)", re.I)
SPATIAL_FEAT = re.compile(r"(neighbo|spatial_?lag|adjacen|adj_|_adj|w_lag|knn_|contig|regic|commut|mobility_lag|"
                          r"global_mean|cross_region)", re.I)
TRANSFER_CLAIM = re.compile(r"(unseen|new|held[- ]?out|left[- ]?out|other)\s+(regions?|areas?|municipalit\w+|"
                            r"locations?|districts?|cities|states?|countries|counties|units?)", re.I)

GROUP_AWARE = {"GroupKFold", "LeaveOneGroupOut", "LeavePGroupsOut", "GroupShuffleSplit",
               "StratifiedGroupKFold", "SpatialKFold", "BlockingTimeSeriesSplit"}
TEMPORAL_SPLITTERS = {"TimeSeriesSplit", "ExpandingWindowSplitter", "SlidingWindowSplitter",
                      "temporal_train_test_split", "BlockingTimeSeriesSplit", "train_test_split_time"}
NON_TEMPORAL_CV = {"KFold", "StratifiedKFold", "ShuffleSplit", "RepeatedKFold", "RepeatedStratifiedKFold",
                   "StratifiedShuffleSplit", "LeaveOneOut"}
CV_CONSUMERS = {"cross_val_score", "cross_validate", "cross_val_predict", "GridSearchCV",
                "RandomizedSearchCV", "BayesSearchCV", "HalvingGridSearchCV", "learning_curve",
                "validation_curve"}


def _f(m: Module, rule: str, node: ast.AST, conf: str, msg: str, pattern: str) -> Finding:
    name, kn, fix = FM_RULES[rule]
    return m.finding(rule, name, node, conf, msg, pattern=pattern, kn=kn, fix=fix)


def _is_temporal(m: Module) -> bool:
    return bool(TIME_SIGNAL.search(m.source)) or bool(m.repo.get("temporal"))


def _assigned_from(m: Module, classes: set[str]) -> set[str]:
    out = set()
    for a in m.assigns:
        if isinstance(a.value, ast.Call) and call_name(a.value) in classes:
            for t in a.targets:
                d = dotted(t)
                if d:
                    out.add(d)
    return out


# ---------------------------------------------------------------------------
# FM1
# ---------------------------------------------------------------------------

def detect_fm1(m: Module) -> list[Finding]:
    out: list[Finding] = []
    if not _is_temporal(m):
        return out
    temporal_cv_names = _assigned_from(m, TEMPORAL_SPLITTERS)
    own_tts = any(isinstance(n, ast.FunctionDef) and n.name == "train_test_split" for n in ast.walk(m.tree))
    sk_tts = not own_tts
    nontemporal_names = _assigned_from(m, NON_TEMPORAL_CV)
    for c in m.calls():
        nm = call_name(c)
        # (a) train_test_split with the default shuffle
        if nm == "train_test_split" and sk_tts and dotted(c.func) in {"train_test_split",
                                                                      "model_selection.train_test_split",
                                                                      "sklearn.model_selection.train_test_split"}:
            sh = kwarg(c, "shuffle")
            if sh is None or (isinstance(sh, ast.Constant) and sh.value is True):
                strat = kwarg(c, "stratify")
                out.append(_f(m, "FM1", c, "high",
                              "train_test_split() shuffles by default, so time-ordered rows are assigned to train "
                              "and test at random: the model is trained on weeks after the ones it is tested on"
                              + (" (stratify= also implies shuffling)." if strat is not None else "."),
                              "random-train-test-split"))
        # (b) non-temporal CV objects on time-indexed data
        elif nm in NON_TEMPORAL_CV and not m.enclosing(c, (ast.ImportFrom,)):
            sh = kwarg(c, "shuffle")
            shuffled = isinstance(sh, ast.Constant) and sh.value is True
            out.append(_f(m, "FM1", c, "high" if shuffled or nm in {"ShuffleSplit", "StratifiedShuffleSplit"} else "medium",
                          f"{nm} on time-indexed data: folds are not ordered in time, so most folds train on "
                          f"periods that come after the fold being predicted"
                          + (" (shuffle=True)." if shuffled else " (even with shuffle=False, earlier folds are "
                             "predicted from later ones)."), "non-temporal-cv"))
        # (c) CV consumers with an integer cv (KFold under the hood) or a non-temporal splitter
        elif nm in CV_CONSUMERS:
            cv = kwarg(c, "cv")
            if cv is None and nm in {"cross_val_score", "cross_validate", "cross_val_predict", "GridSearchCV",
                                     "RandomizedSearchCV"}:
                out.append(_f(m, "FM1", c, "medium",
                              f"{nm}() without cv= uses (Stratified)KFold, which ignores time order.",
                              "default-kfold"))
            elif cv is not None:
                txt = ast.unparse(cv)
                if const_number(cv) is not None or (isinstance(cv, ast.Name) and cv.id in nontemporal_names) or \
                        (isinstance(cv, ast.Call) and call_name(cv) in NON_TEMPORAL_CV):
                    out.append(_f(m, "FM1", c, "medium",
                                  f"{nm}(cv={txt}) uses k-fold splits that ignore time order.", "integer-cv"))
                elif isinstance(cv, ast.Name) and cv.id in temporal_cv_names:
                    pass
        # (d) shuffled Keras validation
        elif nm == "fit" and isinstance(kwarg(c, "validation_split"), ast.Constant):
            pass  # Keras takes the last fraction before shuffling: temporal enough
    # (e) series reversed before a positional split
    for a in m.assigns:
        v = a.value
        rev = (isinstance(v, ast.Subscript) and isinstance(v.slice, ast.Slice) and v.slice.lower is None
               and v.slice.upper is None and const_number(v.slice.step) == -1)
        if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Attribute) and v.value.attr == "iloc" and \
                isinstance(v.slice, ast.Slice) and const_number(v.slice.step) == -1:
            rev = True
        if isinstance(v, ast.Call) and call_name(v) == "sort_values":
            asc = kwarg(v, "ascending")
            by = ast.unparse(kwarg(v, "by") or (v.args[0] if v.args else ast.Constant("")))
            if isinstance(asc, ast.Constant) and asc.value is False and re.fullmatch(
                    r"['\"\[]*(date|ds|time|timestamp|week|week_start|epiweek|year|month|period|dt_\w+)['\"\]]*",
                    by.strip(), re.I):
                rev = True
        if not rev or not a.targets or not isinstance(a.targets[0], ast.Name):
            continue
        name = a.targets[0].id
        if role_ok := (expr_role(v) in ("full",)):
            pass
        # a later positional split of the reversed data
        later = [n for n in ast.walk(m.tree) if isinstance(n, ast.Subscript) and m.line_of(n) > a.line
                 and isinstance(n.slice, ast.Slice) and (n.slice.lower is not None or n.slice.upper is not None)]
        if later and role_ok:
            out.append(_f(m, "FM1", v, "medium",
                          f"'{name}' is put in reverse time order before positional slicing; the first part "
                          f"(training) is then the most recent data and the test set lies in the past.",
                          "reversed-before-positional-split"))
    # (f) reversed data passed straight into a split / window builder: f(df[::-1]) -> X_train, ...
    for n in ast.walk(m.tree):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice) and n.slice.lower is None \
                and n.slice.upper is None and const_number(n.slice.step) == -1 and isinstance(m.parent(n), ast.Call):
            call = m.parent(n)
            stmt = m.enclosing(n, (ast.Assign,))
            if stmt is None or re.search(r"(plot|bar|fill|scatter|list|apply|join|plt|ax\.)", dotted(call.func), re.I):
                continue
            tnames = {x.id for t in stmt.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
            from .core import role_of
            if any(role_of(t) in ("train", "test") for t in tnames):
                out.append(_f(m, "FM1", n, "medium",
                              f"The series is reversed ({ast.unparse(n)}) before being split into "
                              f"{sorted(tnames)[:4]}; the training part is then the most recent data and the "
                              f"test set lies in the past.", "reversed-before-positional-split"))
                break
    return out


# ---------------------------------------------------------------------------
# FM2
# ---------------------------------------------------------------------------

def detect_fm2(m: Module) -> list[Finding]:
    out: list[Finding] = []
    if VINTAGE_GUARD.search(m.source) or m.repo.get("vintage_guard"):
        return out
    evaluates = m.repo.get("evaluates", False) or bool(
        re.search(r"(mean_absolute_error|mean_squared_error|r2_score|\bmae\b|\brmse\b|\bwis\b|crps|train_test_split|"
                  r"backtest|forecast)", m.source, re.I))
    # (a) notification date used as the time anchor
    notif_used = []
    for n in ast.walk(m.tree):
        s = str_const(n) if isinstance(n, ast.Constant) else None
        if s and NOTIF_COLS.fullmatch(s.strip()) or (s and NOTIF_COLS.search(s) and len(s) < 30):
            par = m.parent(n)
            ctx = ast.unparse(m.enclosing(n, (ast.stmt,)) or n)
            if re.search(r"(to_datetime|groupby|resample|dt\.|strftime|value_counts|isocalendar|week|month|year|"
                         r"usecols|sort_values|set_index)", ctx):
                notif_used.append(n)
    if notif_used:
        onset_seen = bool(ONSET_COLS.search(m.source))
        node = notif_used[0]
        if onset_seen:
            out.append(_f(m, "FM2", node, "high",
                          "Cases are dated by notification date although an onset date is available in the data; "
                          "recent periods are incomplete when the forecast is made and no lag correction is applied.",
                          "notification-anchor-onset-available"))
        else:
            out.append(_f(m, "FM2", node, "medium",
                          "Cases are counted by notification date and the most recent periods are used as if "
                          "complete; no reporting-delay correction or data vintage is applied.",
                          "notification-anchor-no-correction"))
    # (b) latest-vintage download of a revisable surveillance source used for evaluation
    for n in ast.walk(m.tree):
        s = str_const(n) if isinstance(n, ast.Constant) else None
        if s and REVISABLE_SOURCES.search(s) and (s.startswith(("http", "ftp")) or "/" in s or "." in s):
            if evaluates:
                out.append(_f(m, "FM2", n, "medium",
                              f"Data come from a source that revises past counts ({REVISABLE_SOURCES.search(s).group(0)}) "
                              f"and are downloaded as the latest snapshot; back-tests and test scores then use values "
                              f"that were not available at each forecast date.", "latest-vintage-revisable-source"))
            break
    # (c) a provisional / partial period labelled as such in a file name and used as data
    for c in m.calls():
        if call_name(c) in {"read_csv", "read_excel", "read_pdf", "read_parquet", "open", "read_table"}:
            lits = [x.value for x in ast.walk(c) if isinstance(x, ast.Constant) and isinstance(x.value, str)]
            if any(re.search(r"(prov|provisional|partial|till|until|ytd|preliminar)", x, re.I) for x in lits):
                out.append(_f(m, "FM2", c, "medium",
                              "A file marked as provisional or partial-period is loaded as data; its last period "
                              "is incomplete and will be revised.", "provisional-file"))
                break
    return out


# ---------------------------------------------------------------------------
# FM3
# ---------------------------------------------------------------------------

def _unit_columns(m: Module) -> set[str]:
    cols = set()
    for n in ast.walk(m.tree):
        s = str_const(n) if isinstance(n, ast.Constant) else None
        if s and UNIT_COLS.match(s):
            cols.add(s)
    return cols


def detect_fm3(m: Module) -> list[Finding]:
    out: list[Finding] = []
    units = _unit_columns(m)
    if not units:
        return out
    names = {call_name(c) for c in m.calls()}
    # (a) units held out, but spatial features built from all units
    unit_holdout = bool(names & GROUP_AWARE) or any(
        isinstance(c, ast.Call) and call_name(c) == "isin" and isinstance(c.func, ast.Attribute)
        and any(str_const(k) in units for k in ast.walk(c.func.value) if isinstance(k, ast.Constant))
        and re.search(r"(test|holdout|val)", ast.unparse(m.enclosing(c, (ast.stmt,)) or c), re.I)
        for c in m.calls())
    if unit_holdout:
        for a in m.assigns:
            tgt = ast.unparse(a.targets[0]) if a.targets else ""
            if SPATIAL_FEAT.search(tgt) and expr_role(a.value) == "full" and \
                    (m.split_line is None or a.line < m.split_line):
                out.append(_f(m, "FM3", a.value, "high",
                              f"Units are held out for testing, but the spatial feature '{tgt[:40]}' is computed "
                              f"from all units, including the held-out ones.", "spatial-feature-from-all-units"))
                break
    # (b) random row split on a multi-unit panel with a stated transfer claim
    claim = TRANSFER_CLAIM.search(m.source) or m.repo.get("transfer_claim")
    if claim and not (names & GROUP_AWARE):
        for c in m.calls():
            if call_name(c) == "train_test_split":
                sh = kwarg(c, "shuffle")
                if sh is None or (isinstance(sh, ast.Constant) and sh.value is True):
                    if kwarg(c, "groups") is None:
                        out.append(_f(m, "FM3", c, "medium",
                                      f"Rows of a multi-unit panel ({sorted(units)[:3]}) are split at random while the "
                                      f"code or docs claim generalisation to other units; the same units appear in "
                                      f"train and test.", "random-split-transfer-claim"))
                        break
    return out


# ---------------------------------------------------------------------------
# FM4
# ---------------------------------------------------------------------------

LOADERS = {"read_csv", "read_excel", "read_parquet", "read_dbc", "read_dbf", "read_stata", "read_sas",
           "read_table", "read_feather", "read_json"}
HARMONISE = re.compile(r"(columns\.intersection|if \w+ in \w+\.columns|\.intersection\(|reindex\(\s*columns|common_cols|common_columns|"
                       r"set\.intersection|&\s*set\(|harmoni[sz]e|rename\(columns\s*=\s*\w*map|COLUMN_MAP|col_map|"
                       r"schema_map|year_map|codebook|recode|map_codes|standardi[sz]e_columns)", re.I)


def detect_fm4(m: Module) -> list[Finding]:
    out: list[Finding] = []
    harmonised = bool(HARMONISE.search(m.source))
    for node in ast.walk(m.tree):
        if not isinstance(node, (ast.For, ast.ListComp, ast.GeneratorExp)):
            continue
        it = node.iter if isinstance(node, ast.For) else node.generators[0].iter
        tgt = node.target if isinstance(node, ast.For) else node.generators[0].target
        it_txt = ast.unparse(it)
        loop_var = names_in(tgt)
        yearish_var = any(re.search(r"(year|ano|yr)", v, re.I) for v in loop_var)
        over_years = yearish_var or bool(re.search(r"(range\(\s*(19|20)\d\d|years?\b|anos?\b)", it_txt, re.I))
        body = node.body if isinstance(node, ast.For) else [node.elt]
        loads = [c for b in body for c in ast.walk(b) if isinstance(c, ast.Call) and call_name(c) in LOADERS]
        if not loads:
            continue
        load = loads[0]
        uses_var = bool(names_in(load) & loop_var)
        # a glob/list of files counts when the file pattern itself is year-keyed (DENGBR*, sinan_2015.csv)
        if not over_years:
            pat = " ".join(x.value for x in ast.walk(m.tree) if isinstance(x, ast.Constant) and isinstance(x.value, str))
            over_years = bool(re.search(r"(DENG|CHIK|ZIKA|SINAN)[A-Z]{0,2}\*|\{year\}|_\*\d{0,2}\.csv", pat, re.I)) \
                and bool(re.search(r"(glob|listdir|iterdir|files)", it_txt, re.I))
        if not uses_var or not over_years:
            continue
        usecols = kwarg(load, "usecols") is not None or kwarg(load, "columns") is not None
        if harmonised:
            continue
        if usecols:
            # usecols fixes the columns read, not their meaning across years
            conf, extra = "low", " usecols= fixes which columns are read, but not whether codes or definitions " \
                                 "changed between years."
        else:
            conf, extra = "medium", " No column intersection, mapping or usecols restriction follows."
        out.append(_f(m, "FM4", load, conf,
                      f"Files for several years are loaded in a loop over '{it_txt[:40]}' and combined without "
                      f"harmonising schemas or codes across years." + extra, "multi-year-load-unharmonised"))
        break
    # pd.concat of year-keyed reads written out explicitly
    if not out and not harmonised:
        for c in m.calls():
            if call_name(c) == "concat" and c.args and isinstance(c.args[0], (ast.List, ast.Tuple)):
                reads = [e for e in c.args[0].elts if isinstance(e, ast.Call) and call_name(e) in LOADERS]
                yrs = [e for e in reads if re.search(r"(19|20)\d\d", ast.unparse(e))]
                if len(yrs) >= 2:
                    out.append(_f(m, "FM4", c, "medium",
                                  "Year-specific files are concatenated without harmonising columns or codes.",
                                  "concat-year-files"))
                    break
    return out


FM_DETECTORS = {"FM1": detect_fm1, "FM2": detect_fm2, "FM3": detect_fm3, "FM4": detect_fm4}
