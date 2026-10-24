"""Detectors for the candidate failure modes of the derivation set.

One function per candidate. Each takes a `Module` and returns `Finding`s.
They are heuristics over the AST plus a few cheap facts about the data
(CSV headers next to the code, when present). Confidence levels mean:

    high    the pattern itself is the failure (e.g. a scaler fitted on X_test)
    medium  the pattern is the failure unless something not visible to static
            analysis rules it out (e.g. same-week weather that is itself forecast)
    low     a code shape that often carries the failure; worth a look

Candidates covered: CL-05, CL-06, CL-07, CL-09, CL-10, CL-11. FM1-FM4 are in
fm_detectors.py. CL-01-CL-04 (need the data calendar or the authors' claims) and
single-occurrence candidates (CL-08, CL-12-CL-14) are not implemented.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .core import (Finding, Module, call_name, columns_referenced, const_number, dotted,
                   expr_role, first_arg, kwarg, method_chain, names_in, role_of, shift_sign,
                   str_const, subscript_key, FIT_CLASSES, LAG_WORD_RE, METRIC_FUNCS, WEATHER_RE)

RULES = {
    "CL-05": ("Covariates aligned to future weeks", "L3.1",
              "Lag every covariate by at least the forecast horizon, or use values that "
              "are themselves forecasts available at the forecast origin."),
    "CL-06": ("Horizon counted from data week, not forecast availability", "L3.1",
              "Start the target window after the last input period, and count the horizon "
              "from when the forecast can be issued."),
    "CL-07": ("Preprocessing statistics from evaluation data", "L1.2",
              "Fit scalers, imputers and normalisation statistics on training rows only, "
              "then apply them to validation and test rows."),
    "CL-09": ("Best model chosen on the test set", "L1.1",
              "Select models, epochs, orders and features on a validation split or inner "
              "CV; touch the test set once."),
    "CL-10": ("Outbreak threshold set from the evaluation data", "L1.2",
              "Fix outbreak or alert thresholds from training data (or an external rule) "
              "before scoring."),
    "CL-11": ("Feature computed from the same-period target", "L2",
              "Shift the target before any rolling, ratio or normalisation used as a feature "
              "(e.g. df[y].shift(1).rolling(k))."),
}

ROLLING_OPS = {"rolling", "ewm", "expanding", "cumsum", "cummax", "cummin", "cumprod",
               "pct_change", "diff"}
RESHAPE_GRID = {"reindex", "asfreq", "resample", "date_range", "period_range", "complete",
                "fill_missing_dates", "set_index_complete", "MultiIndex"}
THRESH_NAME = re.compile(r"(thresh|^thr$|thr_|^tau|cutoff|cut_off|outbreak|epidemic|alert|surge|"
                         r"alarm|pos_threshold)", re.I)
CLS_METRICS = {"f1_score", "precision_score", "recall_score", "roc_auc_score", "accuracy_score",
               "average_precision_score", "confusion_matrix", "classification_report",
               "precision_recall_curve", "roc_curve", "balanced_accuracy_score", "matthews_corrcoef"}
LOOP_ORIGIN = re.compile(r"(origin|fold|cutoff|window|split|year|season|step|week|month|"
                         r"start|end|t0|anchor|period|train_end|horizon_start)", re.I)


def _f(m: Module, rule: str, node: ast.AST, conf: str, msg: str, pattern: str) -> Finding:
    name, kn, fix = RULES[rule]
    return m.finding(rule, name, node, conf, msg, pattern=pattern, kn=kn, fix=fix)


def _calls_in_chain(node: ast.AST) -> tuple[ast.AST, list[ast.Call]]:
    base, chain = method_chain(node)
    return base, [c for c in chain if isinstance(c, ast.Call)]


def _is_chain_root(m: Module, node: ast.AST) -> bool:
    """False when node is an inner link of a longer method chain (x.a().b(): x.a() is not root)."""
    par = m.parent(node)
    if isinstance(par, ast.Attribute):
        gp = m.parent(par)
        if isinstance(gp, ast.Call) and gp.func is par:
            return False
    if isinstance(par, ast.Subscript) and par.value is node:
        gp = m.parent(par)
        if isinstance(gp, ast.Attribute):
            return False
    return True


def _chain_refs_target(m: Module, base: ast.AST, chain: list) -> bool:
    if subscript_key(base) in m.targets:
        return True
    for c in chain:
        if isinstance(c, ast.Subscript) and subscript_key(c) in m.targets:
            return True
    return False


def _assigned_column(stmt: ast.AST) -> str | None:
    """df['new'] = ...  ->  'new'  (also .alias('new') in polars chains)"""
    if isinstance(stmt, ast.Assign):
        for n in ast.walk(stmt.value):
            if isinstance(n, ast.Call) and call_name(n) == "alias" and n.args and str_const(n.args[0]):
                return str_const(n.args[0])
    if isinstance(stmt, ast.Assign) and stmt.targets:
        t = stmt.targets[0]
        if isinstance(t, ast.Subscript):
            return subscript_key(t)
        if isinstance(t, ast.Name):
            return t.id
    return None


def _used_as_feature(m: Module, col: str | None) -> bool:
    """The new column name appears in a literal list/tuple, or X is 'all columns but the target'."""
    if not col:
        return False
    if any(col in names for _, names in _feature_lists(m)):
        return True
    if not any(call_name(c) in {"fit", "fit_transform", "train", "fit_predict"} for c in m.calls()):
        return False
    for c in m.calls():
        if call_name(c) == "drop" and m.refs_target(c):
            return True
        if call_name(c) == "select_dtypes":
            return True
    for n in ast.walk(m.tree):
        if isinstance(n, ast.ListComp) and "columns" in ast.unparse(n.generators[0].iter):
            return True
    return False


def _windowed(m: Module) -> bool:
    """Sliding-window sample construction: x = a[i:i+w], y = a[i+w] (inputs precede the target)."""
    for n in ast.walk(m.tree):
        if isinstance(n, ast.Slice) and isinstance(n.upper, ast.BinOp) and isinstance(n.upper.op, ast.Add) \
                and n.lower is not None and names_in(n.lower) & names_in(n.upper):
            return True
    return False


def _is_targetish(name: str | None, m: Module) -> bool:
    if not name:
        return False
    return name in m.targets or bool(re.search(r"(target|label|^y$|^y_|_y$|outcome|future|ahead|lead)",
                                               name, re.I))


# ---------------------------------------------------------------------------
# CL-11  Feature computed from the same-period target
# ---------------------------------------------------------------------------

def detect_cl11(m: Module) -> list[Finding]:
    out: list[Finding] = []
    if not m.targets:
        return out
    seen: set[int] = set()
    for stmt in ast.walk(m.tree):
        if not isinstance(stmt, ast.Assign):
            continue
        newcol = _assigned_column(stmt)
        if _is_targetish(newcol, m):
            continue
        for node in ast.walk(stmt.value):
            if not isinstance(node, ast.Call) or id(node) in seen or not _is_chain_root(m, node):
                continue
            base, chain = method_chain(node)
            calls = [c for c in chain if isinstance(c, ast.Call)]
            ops = [call_name(c) for c in calls]
            if not _chain_refs_target(m, base, chain):
                continue
            shifts = [shift_sign(c) for c in calls if call_name(c) == "shift"]
            if any(s != 0 for s in shifts):
                continue  # lagged (fine) or a future target (CL-06/CL-05 territory)
            # transform(lambda s: s.rolling(k).mean()) without shift inside
            lam_roll = False
            for c in calls:
                if call_name(c) in {"transform", "apply"}:
                    for a in c.args:
                        if isinstance(a, ast.Lambda):
                            inner = {call_name(x) for x in ast.walk(a.body) if isinstance(x, ast.Call)}
                            if inner & ROLLING_OPS and "shift" not in inner:
                                lam_roll = True
            if set(ops) & ROLLING_OPS or lam_roll:
                for c in calls:
                    seen.add(id(c))
                op = next((o for o in ops if o in ROLLING_OPS), "rolling")
                if op == "diff" and re.search(r"(new|daily|weekly|incid|^num|_inc$)", newcol or "", re.I):
                    break  # differencing cumulative counts to build the outcome itself
                conf = "high" if _used_as_feature(m, newcol) else "low"
                out.append(_f(m, "CL-11", node, conf,
                              f"'{newcol or 'feature'}' applies {op}() to the target column "
                              f"{sorted(columns_referenced(node) & m.targets)} with no shift, so it "
                              f"includes the value being predicted.", "unshifted-rolling-on-target"))
                break
        else:
            # arithmetic on the bare target: ratio, z-score, per-capita of the same period
            if newcol and isinstance(stmt.value, ast.BinOp):
                bare = [s for s in ast.walk(stmt.value)
                        if isinstance(s, ast.Subscript) and subscript_key(s) in m.targets
                        and not isinstance(m.parent(s), ast.Attribute)]
                has_shift = any(isinstance(c, ast.Call) and call_name(c) == "shift"
                                for c in ast.walk(stmt.value))
                other = [n for n in ast.walk(stmt.value)
                         if isinstance(n, (ast.Subscript, ast.Name, ast.Call))
                         and not (isinstance(n, ast.Subscript) and subscript_key(n) in m.targets)]
                if bare and not has_shift and other and _used_as_feature(m, newcol) and \
                        not re.search(r"log|sqrt|scaled|norm_target|resid|error|err$|diff$", newcol, re.I):
                    out.append(_f(m, "CL-11", stmt, "medium",
                                  f"'{newcol}' is an arithmetic function of the same-period target "
                                  f"{sorted({subscript_key(b) for b in bare})}; if it is used as an input, "
                                  f"the label leaks into the features.", "target-arithmetic"))
    return out


# ---------------------------------------------------------------------------
# CL-06  Horizon counted from the data period
# ---------------------------------------------------------------------------

def _one_minus(arg: ast.AST | None) -> bool:
    """shift(1 - h), shift(-(h - 1)), shift(-h + 1)."""
    if arg is None:
        return False
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Sub) and const_number(arg.left) == 1:
        return True
    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) and isinstance(arg.operand, ast.BinOp) \
            and isinstance(arg.operand.op, ast.Sub) and const_number(arg.operand.right) == 1:
        return True
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add) and const_number(arg.right) == 1 \
            and isinstance(arg.left, ast.UnaryOp):
        return True
    return False


def detect_cl06(m: Module) -> list[Finding]:
    out: list[Finding] = []
    for c in m.calls():
        if call_name(c) != "shift":
            continue
        base, chain = method_chain(c)
        ops = [call_name(x) for x in chain if isinstance(x, ast.Call)]
        if "rolling" not in ops:
            continue
        arg = first_arg(c, "periods")
        v = const_number(arg)
        if _one_minus(arg) or (v is not None and v < 0 and ops.index("rolling") < ops.index("shift")):
            win = None
            for x in chain:
                if isinstance(x, ast.Call) and call_name(x) == "rolling":
                    win = first_arg(x, "window")
            same = (v is not None and win is not None and const_number(win) is not None
                    and v == 1 - const_number(win))
            if _one_minus(arg) or same:
                out.append(_f(m, "CL-06", c, "medium",
                              "The target is a rolling sum shifted by 1 - window, so the target window "
                              "starts at the current row, which is also the last input row: the first "
                              "period of the 'h-ahead' target is already observed.",
                              "rolling-target-starts-at-t"))
    return out


# ---------------------------------------------------------------------------
# CL-05  Covariates aligned to future periods
# ---------------------------------------------------------------------------

_HEADER_CACHE: dict[str, set[str]] = {}


def _read_header(f: Path) -> set[str]:
    key = str(f)
    if key not in _HEADER_CACHE:
        cols: set[str] = set()
        try:
            with f.open(encoding="utf-8", errors="replace") as fh:
                head = fh.readline()
            for sep in (",", ";", "\t"):
                if head.count(sep) >= 1:
                    cols = {h.strip().strip('"').strip("'") for h in head.split(sep)}
                    break
        except OSError:
            pass
        _HEADER_CACHE[key] = cols
    return _HEADER_CACHE[key]


def _csv_columns(m: Module) -> set[str]:
    """Headers of CSV files the code reads. Paths built with os.path.join or Path / '...' are
    resolved by file name inside the scanned repository."""
    cols: set[str] = set()
    root = Path(m.root) if m.root else Path(m.path).parent
    for c in m.calls():
        if call_name(c) not in {"read_csv", "read_table"} or not c.args:
            continue
        exprs = [c.args[0]]
        # read_csv(DATA_PATH): follow the name to its assignment
        for nm_ in names_in(c.args[0]):
            exprs += [a.value for a in m.assigns for t in a.targets if isinstance(t, ast.Name) and t.id == nm_]
        lits = [n.value for e in exprs for n in ast.walk(e) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        names = [Path(x).name for x in lits if x.endswith((".csv", ".tsv", ".txt"))]
        for name in names:
            for f in list(root.rglob(name))[:3]:
                if f.is_file() and f.stat().st_size < 500_000_000:
                    cols |= _read_header(f)
    return cols


def _feature_lists(m: Module) -> list[tuple[ast.AST, list[str]]]:
    """Literal lists of column names that are used as model inputs."""
    res = []
    for a in m.assigns:
        for t in a.targets:
            tname = t.id if isinstance(t, ast.Name) else dotted(t)
            if not tname:
                continue
            is_feat_name = re.search(r"(feature|feat|cols|columns|predictors|regressors|exog|covariates|"
                                     r"^x$|^x_)", tname, re.I) and \
                not re.search(r"(required|expected|climate_col|weather_col|raw|base_|_base|numeric|dtype|"
                              r"rename|drop|keep_cols|id_cols|meta|ocean|enso)", tname, re.I)
            v = a.value
            if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add):
                elts = []
                for side in (v.left, v.right):
                    if isinstance(side, (ast.List, ast.Tuple)):
                        elts += side.elts
            elif isinstance(v, (ast.List, ast.Tuple)):
                elts = list(v.elts)
            elif isinstance(v, ast.Subscript) and isinstance(v.slice, (ast.List, ast.Tuple)):
                elts = list(v.slice.elts)  # X = df[[...]]
                is_feat_name = is_feat_name or re.match(r"^x", tname, re.I)
            else:
                continue
            names = [str_const(e) for e in elts if str_const(e)]
            if is_feat_name and names:
                res.append((t, names))  # type: ignore[arg-type]
    return res


def _target_is_future(m: Module) -> bool:
    """True if the code builds a target by shifting the outcome forward (shift(-h))."""
    for c in m.calls():
        if call_name(c) == "shift" and shift_sign(c) < 0:
            base, chain = method_chain(c)
            stmt = m.enclosing(c, (ast.Assign,))
            if _chain_refs_target(m, base, chain) or _is_targetish(_assigned_column(stmt) if stmt else None, m):
                return True
    return False


def detect_cl05(m: Module) -> list[Finding]:
    out: list[Finding] = []
    # (a) covariates shifted backwards (future values brought to row t)
    for c in m.calls():
        if call_name(c) != "shift" or shift_sign(c) >= 0:
            continue
        base, chain = method_chain(c)
        if _chain_refs_target(m, base, chain):
            continue  # that is target construction
        stmt = m.enclosing(c, (ast.Assign,))
        newcol = _assigned_column(stmt) if stmt else None
        if _is_targetish(newcol, m):
            continue
        cols = sorted(columns_referenced(c) - m.targets)
        if not cols and not newcol:
            continue
        out.append(_f(m, "CL-05", c, "high",
                      f"Covariate {cols or [newcol]} is shifted backwards (shift with a negative "
                      f"period), so row t holds a future value.", "negative-shift-covariate"))

    # (b) same-period exogenous covariates with a same-period target
    future_target = _target_is_future(m)
    lagged_lists: set[str] = set()
    for node in ast.walk(m.tree):  # for c in FEATURES: df[c + '_lag'] = df[c].shift(k)
        if isinstance(node, ast.For) and any(isinstance(c, ast.Call) and call_name(c) == "shift"
                                             for c in ast.walk(node)):
            lagged_lists |= names_in(node.iter)
    if not future_target and m.targets and not _windowed(m):
        flagged: set[str] = set()
        for node, names in _feature_lists(m):
            if dotted(node) in lagged_lists:
                continue
            same = [n for n in names if WEATHER_RE.search(n) and not LAG_WORD_RE.search(n)
                    and n not in m.targets]
            if same:
                flagged |= set(same)
                out.append(_f(m, "CL-05", node, "medium",
                              f"Inputs {same[:8]} are covariates of the same period as the target "
                              f"{sorted(m.targets)} (no lag, and the target is not shifted forward). They "
                              f"are not known when the forecast is made unless they are themselves forecasts.",
                              "same-period-covariates"))
        # X = df.drop(columns=[target, ...]) or comprehension over df.columns: use CSV header
        if not flagged:
            header = _csv_columns(m)
            same = sorted(h for h in header if WEATHER_RE.search(h) and not LAG_WORD_RE.search(h)
                          and h not in m.targets)
            if same:
                for c in m.calls():
                    if call_name(c) == "drop" and m.refs_target(c):
                        stmt = m.enclosing(c, (ast.Assign,))
                        tn = _assigned_column(stmt) if stmt else ""
                        if tn and re.match(r"^x", tn, re.I):
                            out.append(_f(m, "CL-05", c, "medium",
                                          f"X is every column except the target; the data file has "
                                          f"same-period covariates {same[:8]} with no lag.",
                                          "same-period-covariates-from-data"))
                            break
                else:
                    for node in ast.walk(m.tree):
                        if isinstance(node, ast.ListComp) and "columns" in ast.unparse(node.generators[0].iter) \
                                and any(str_const(e) in m.targets for e in ast.walk(node) if str_const(e)):
                            out.append(_f(m, "CL-05", node, "medium",
                                          f"Features are all data columns except the target; the data "
                                          f"file has same-period covariates {same[:8]} with no lag.",
                                          "same-period-covariates-from-data"))
                            break

    # (c) lag search that admits lag 0 for covariates
    for node in ast.walk(m.tree):
        if isinstance(node, (ast.For, ast.comprehension)):
            tgt = node.target
            it = node.iter
            starts_at_zero = isinstance(it, ast.Call) and call_name(it) == "range" and it.args and (
                len(it.args) == 1 or const_number(it.args[0]) == 0)
            fits_here = any(call_name(c) in {"fit", "predict", "fit_predict", "train"} for c in m.calls())
            if isinstance(tgt, ast.Name) and "lag" in tgt.id.lower() and starts_at_zero and fits_here:
                scope = m.enclosing(node, (ast.FunctionDef,)) or node
                covariate = bool(WEATHER_RE.search(ast.unparse(scope))) and isinstance(node, ast.For) and \
                    any(isinstance(c, ast.Call) and call_name(c) in {"shift", "corr", "fit", "roll"}
                        for c in ast.walk(node))
                out.append(_f(m, "CL-05", node if isinstance(node, ast.For) else it, "medium" if covariate else "low",
                              "A lag search for covariates includes lag 0, so the same period's (unknown at "
                              "forecast time) covariate values can be selected." if covariate else
                              "A lag search starts at 0, so same-period covariates can be selected.",
                              "lag-search-includes-zero"))
    return out


# ---------------------------------------------------------------------------
# CL-07  Preprocessing statistics from evaluation data
# ---------------------------------------------------------------------------

def _is_encoder(m: Module, recv: ast.AST) -> bool:
    """Label/one-hot/ordinal encoders only learn the category set; fitting them before the split
    does not carry evaluation-period statistics in any meaningful way."""
    if isinstance(recv, ast.Call):
        return "Encoder" in call_name(recv) or "Binarizer" in call_name(recv)
    name = dotted(recv)
    for a in m.assigns:
        if isinstance(a.value, ast.Call) and any(dotted(t) == name for t in a.targets):
            cn = call_name(a.value)
            if "Encoder" in cn or "Binarizer" in cn:
                return True
    return False


def _fitter_names(m: Module) -> set[str]:
    names = set()
    for a in m.assigns:
        if isinstance(a.value, ast.Call) and FIT_CLASSES.search(call_name(a.value) or ""):
            for t in a.targets:
                d = dotted(t)
                if d:
                    names.add(d)
    return names


def detect_cl07(m: Module) -> list[Finding]:
    out: list[Finding] = []
    fitters = _fitter_names(m)
    models_here = any(call_name(c) in {"fit", "fit_transform", "train", "predict", "fit_predict", "DataLoader",
                                       "train_test_split", "SARIMAX", "ARIMA", "run", "minimize"}
                      for c in m.calls()) or m.split_line is not None or any(
        isinstance(n, ast.ClassDef) and any(re.search(r"Dataset|DataModule|Loader", ast.unparse(b)) for b in n.bases)
        for n in ast.walk(m.tree))
    for c in m.calls():
        nm = call_name(c)
        if nm in {"fit", "fit_transform", "partial_fit"} and isinstance(c.func, ast.Attribute):
            recv = c.func.value
            is_fitter = dotted(recv) in fitters or (isinstance(recv, ast.Call)
                                                     and FIT_CLASSES.search(call_name(recv) or ""))
            if not is_fitter or not c.args:
                continue
            role = expr_role(c.args[0])
            arg_txt = ast.unparse(c.args[0])[:60]
            if role == "test":
                out.append(_f(m, "CL-07", c, "high",
                              f"{dotted(recv) or call_name(recv)}.{nm}() is fitted on test data ({arg_txt}).",
                              "fit-on-test"))
            elif role == "val":
                out.append(_f(m, "CL-07", c, "medium",
                              f"{dotted(recv) or call_name(recv)}.{nm}() is fitted on validation data "
                              f"({arg_txt}), which is then used for evaluation or model selection.",
                              "fit-on-validation"))
            elif role == "full" and m.before_split(c) and not _is_encoder(m, recv):
                out.append(_f(m, "CL-07", c, "medium",
                              f"{dotted(recv) or call_name(recv)}.{nm}() is fitted on {arg_txt} before the "
                              f"data is split (first split at line {m.split_line}), so its statistics include "
                              f"the evaluation period.", "fit-before-split"))
        # imputation from full-series statistics
        elif nm == "fillna" and m.before_split(c):
            arg = first_arg(c, "value")
            if isinstance(arg, ast.Call) and call_name(arg) in {"mean", "median", "mode"} \
                    and expr_role(arg) == "full":
                out.append(_f(m, "CL-07", c, "medium",
                              "Missing values are filled with a statistic of the full series before the "
                              "split.", "fillna-full-stat"))
        elif nm in {"bfill", "backfill"} or (nm == "fillna" and str_const(kwarg(c, "method")) in {"bfill", "backfill"}):
            conf = "low"
            out.append(_f(m, "CL-07", c, conf,
                          "Backward fill copies later values into earlier rows; before the split it "
                          "moves evaluation-period values into training rows.", "backward-fill"))
        elif nm == "interpolate" and str_const(kwarg(c, "limit_direction")) == "both" and \
                (m.before_split(c) or m.split_line is None):
            out.append(_f(m, "CL-07", c, "medium",
                          "interpolate(limit_direction='both') over the whole series fills gaps from "
                          "later values, including the evaluation period.", "interpolate-both"))
        # numpy / pandas normalisation over the full array
        elif nm in {"max", "min", "mean", "std", "nanmax", "nanmin", "nanmean", "nanstd"} and models_here:
            base, chain = method_chain(c)
            if {call_name(x) for x in chain if isinstance(x, ast.Call)} & {"rolling", "ewm", "expanding",
                                                                            "groupby", "resample", "to_datetime"}:
                continue
            is_np = dotted(c.func).startswith(("np.", "numpy."))
            data = c.args[0] if (is_np and c.args) else (c.func.value if isinstance(c.func, ast.Attribute) else None)
            if data is None:
                continue
            drole = expr_role(data)
            if drole in ("test", "val"):
                par = m.parent(c)
                while par is not None and not isinstance(par, ast.stmt):
                    if isinstance(par, ast.BinOp) and isinstance(par.op, (ast.Div, ast.Sub)) and \
                            ast.unparse(data) in ast.unparse(par.left):
                        out.append(_f(m, "CL-07", c, "high",
                                      f"'{ast.unparse(data)[:40]}' is normalised with its own {nm}(), so the "
                                      f"evaluation rows are scaled with statistics of the evaluation period.",
                                      "eval-normalised-with-own-stats"))
                        break
                    par = m.parent(par)
                continue
            if drole != "full":
                continue
            fn = m.enclosing(c, (ast.FunctionDef,))
            in_norm_fn = fn is not None and re.search(r"(normali|scal|standardi|minmax|zscore)", fn.name, re.I)
            axis_given = kwarg(c, "axis") is not None or (is_np and len(c.args) >= 2)
            # (x - x.min()) / (x.max() - x.min())  or  (x - x.mean()) / x.std()  on the same expression
            binop = m.parent(c)
            while binop is not None and isinstance(binop, (ast.BinOp, ast.Call, ast.Attribute)) \
                    and not isinstance(binop, ast.stmt):
                if isinstance(binop, ast.BinOp) and isinstance(binop.op, ast.Div):
                    break
                binop = m.parent(binop)
            same_expr = isinstance(binop, ast.BinOp) and isinstance(binop.op, ast.Div) and \
                ast.unparse(data) in ast.unparse(binop.left).replace(ast.unparse(c), "")
            if (in_norm_fn and (axis_given or nm in {"max", "min", "nanmax", "nanmin"})) or \
                    (same_expr and m.before_split(c)):
                out.append(_f(m, "CL-07", c, "medium",
                              f"Normalisation statistic {nm}() is computed on '{ast.unparse(data)[:40]}', "
                              f"which is not restricted to training rows.", "normalisation-stat-full"))
    # imputation from whole-array statistics: col_means = np.nanmean(exog, axis=0); exog[r, c] = col_means[c]
    fill_stats: dict[str, str] = {}
    for a in m.assigns:
        v = a.value
        if isinstance(v, ast.Call) and call_name(v) in {"nanmean", "nanmedian", "mean", "median"} and \
                dotted(v.func).startswith(("np.", "numpy.")) and v.args and expr_role(v.args[0]) == "full":
            for t in a.targets:
                if isinstance(t, ast.Name):
                    fill_stats[t.id] = ast.unparse(v.args[0])
    if fill_stats and models_here:
        for a in m.assigns:
            for t in a.targets:
                if isinstance(t, ast.Subscript) and names_in(a.value) & set(fill_stats):
                    src = fill_stats[sorted(names_in(a.value) & set(fill_stats))[0]]
                    if ast.unparse(t.value) == src:
                        out.append(_f(m, "CL-07", a.value, "medium",
                                      f"Missing values in '{src}' are filled with its own column statistics over the "
                                      f"whole array, evaluation period included.", "impute-full-array-stats"))
    # two-statement form: mean_test = np.mean(data_test); z = (data_test - mean_test) / std_test
    stat_names: dict[str, ast.AST] = {}
    for a in m.assigns:
        v = a.value
        if isinstance(v, ast.Call) and call_name(v) in {"mean", "std", "min", "max", "nanmean", "nanstd"}:
            is_np = dotted(v.func).startswith(("np.", "numpy."))
            data = v.args[0] if (is_np and v.args) else (v.func.value if isinstance(v.func, ast.Attribute) else None)
            if data is not None and expr_role(data) in ("test", "val"):
                for t in a.targets:
                    if isinstance(t, ast.Name):
                        stat_names[t.id] = data
    for a in m.assigns:
        for n in ast.walk(a.value):
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Sub, ast.Div)):
                used = names_in(n) & set(stat_names)
                if used:
                    d = stat_names[sorted(used)[0]]
                    if ast.unparse(d) in ast.unparse(n):
                        out.append(_f(m, "CL-07", n, "high",
                                      f"'{ast.unparse(d)[:40]}' is normalised with statistics computed on itself "
                                      f"({', '.join(sorted(used))}), i.e. on the evaluation period.",
                                      "eval-normalised-with-own-stats"))
                        break
    # one finding per line and pattern
    uniq, keys = [], set()
    for f in out:
        k = (f.line, f.pattern)
        if k not in keys:
            keys.add(k)
            uniq.append(f)
    return uniq


# ---------------------------------------------------------------------------
# CL-09  Best model chosen on the test set
# ---------------------------------------------------------------------------

def _is_metric_call(c: ast.Call) -> bool:
    nm = call_name(c).lower()
    return nm in METRIC_FUNCS or nm in {"criterion", "loss_fn", "compute_metrics", "evaluate_metrics",
                                        "calc_metrics", "metrics"} \
        or bool(re.search(r"(mae|rmse|mse|mape|smape|wis|score|error|loss|r2|metrics)$", nm))


METRIC_KEY = re.compile(r"(mae|rmse|mse|mape|smape|wis|r2|error|loss|score)", re.I)


def detect_cl09(m: Module) -> list[Finding]:
    out: list[Finding] = []
    # (a) early stopping / eval sets on the test split
    for c in m.calls():
        for kw in ("validation_data", "eval_set", "valid_sets", "X_val", "validation_split_data",
                   "val_dataloaders", "eval_dataset"):
            v = kwarg(c, kw)
            if v is not None and expr_role(v) == "test":
                out.append(_f(m, "CL-09", c, "high",
                              f"{call_name(c)}({kw}=...) uses the test split for monitoring or early "
                              f"stopping, so training choices are made on the test set.", "early-stop-on-test"))
    # (b) a container of test scores that is ranked
    containers: dict[str, ast.AST] = {}
    scored_names: dict[str, ast.AST] = {}
    for c in m.calls():
        if not _is_metric_call(c):
            continue
        if not any(expr_role(a) == "test" for a in list(c.args) + [k.value for k in c.keywords]):
            continue
        stmt = m.enclosing(c, (ast.stmt,))
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Subscript):
                    containers[dotted(t.value)] = c
                elif isinstance(t, ast.Name):
                    scored_names[t.id] = c
            if isinstance(stmt.value, ast.Dict):
                for t in stmt.targets:
                    containers[dotted(t)] = c
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) \
                and call_name(stmt.value) in {"append", "update", "extend"}:
            containers[dotted(stmt.value.func.value)] = c
        # appended as part of a dict: results.append({'model': n, 'mae': mae(y_test, p)})
        par = m.parent(c)
        while par is not None and not isinstance(par, ast.stmt):
            if isinstance(par, ast.Call) and call_name(par) == "append":
                containers[dotted(par.func.value)] = c
            par = m.parent(par)
    # propagate test-score names through simple assignments: rmse = np.sqrt(test_loss)
    for _ in range(2):
        for a in m.assigns:
            if names_in(a.value) & set(scored_names):
                for t in a.targets:
                    if isinstance(t, ast.Name):
                        scored_names.setdefault(t.id, a.value)
    for c in m.calls():
        if call_name(c) in {"append", "extend"} and isinstance(c.func, ast.Attribute) and c.args \
                and names_in(c.args[0]) & set(scored_names):
            containers[dotted(c.func.value)] = c
    # names that receive a test score feed containers too: scores[name] = mae
    for a in m.assigns:
        for t in a.targets:
            if isinstance(t, ast.Subscript) and names_in(a.value) & set(scored_names):
                containers[dotted(t.value)] = a.value
    for c in m.calls():
        nm = call_name(c)
        if nm in {"min", "max", "sorted", "idxmin", "idxmax", "argmin", "argmax", "nsmallest", "nlargest",
                  "sort_values", "sort"}:
            refs = names_in(c) | ({dotted(c.func.value)} if isinstance(c.func, ast.Attribute) else set())
            hit = [k for k in containers if k and (k in refs or k.split(".")[0] in refs)]
            if hit:
                out.append(_f(m, "CL-09", c, "medium",
                              f"{nm}() ranks '{hit[0]}', which holds scores computed on the test set; the "
                              f"selected model is then reported with the same test score.",
                              "rank-test-scores"))
    # (b2) selection keyed on an error metric, in a module that scores the test split and never
    #      scores a validation split
    alias: dict[str, str] = {}
    for a in m.assigns:
        r = expr_role(a.value)
        if r in ("test", "val"):
            for t in a.targets:
                if isinstance(t, ast.Name) and role_of(t.id) == "full":
                    alias[t.id] = r

    def _role(node: ast.AST) -> str:
        r = expr_role(node)
        if r != "full":
            return r
        rs = {alias[n] for n in names_in(node) if n in alias}
        return "test" if "test" in rs else ("val" if "val" in rs else "full")

    def _scores_on(role: str) -> bool:
        for c in m.calls():
            if _is_metric_call(c) and any(_role(a) == role for a in list(c.args) + [k.value for k in c.keywords]):
                return True
        return False
    if not out and _scores_on("test") and not _scores_on("val"):
        for c in m.calls():
            nm = call_name(c)
            keyed = False
            if nm in {"min", "max", "sorted"}:
                k = kwarg(c, "key")
                keyed = k is not None and bool(METRIC_KEY.search(ast.unparse(k)))
            elif nm in {"idxmin", "idxmax", "argmin", "argmax"} and isinstance(c.func, ast.Attribute):
                keyed = bool(METRIC_KEY.search(ast.unparse(c.func.value)))
            elif nm in {"nsmallest", "nlargest"} and m.parent(c) is not None and \
                    isinstance(m.parent(c), ast.Attribute) and m.parent(c).attr in {"index", "iloc"}:
                keyed = bool(METRIC_KEY.search(ast.unparse(c)))
            if keyed:
                out.append(_f(m, "CL-09", c, "medium",
                              f"The best model is picked with {nm}() on an error metric, and the only scores this "
                              f"module computes are on the test/hold-out split; the winner's score is then an "
                              f"optimistic estimate.", "select-by-test-metric"))
                break
    # (c) keep-the-best loops on a test score
    test_only = _scores_on("test") and not _scores_on("val")
    for node in ast.walk(m.tree):
        if isinstance(node, ast.If) and isinstance(node.test, (ast.Compare, ast.BoolOp)):
            used = names_in(node.test)
            hit = used & set(scored_names)
            txt = ast.unparse(node.test)
            if not hit and test_only and re.search(r"best", txt, re.I) and METRIC_KEY.search(txt) \
                    and re.search(r"<|>", txt) and not any(role_of(n) == "val" for n in used):
                hit = {METRIC_KEY.search(txt).group(0)}
            if hit and any(re.search(r"best|min|lowest", n, re.I) for n in used):
                out.append(_f(m, "CL-09", node, "medium",
                              f"A 'best so far' comparison uses {sorted(hit)}, a score on the test set.",
                              "keep-best-on-test"))
    # (d) model order / hyper-parameter search on the full series before the split
    for c in m.calls():
        nm = call_name(c)
        if nm in {"auto_arima", "AutoARIMA"} and c.args and expr_role(c.args[0]) == "full" \
                and m.before_split(c):
            out.append(_f(m, "CL-09", c, "medium",
                          "auto_arima selects the model order on the full series, test period included.",
                          "order-search-full-series"))
        if nm in {"GridSearchCV", "RandomizedSearchCV", "BayesSearchCV"}:
            stmt = m.enclosing(c, (ast.Assign,))
            gs = dotted(stmt.targets[0]) if stmt else ""
            for c2 in m.calls():
                if call_name(c2) == "fit" and isinstance(c2.func, ast.Attribute) and dotted(c2.func.value) == gs \
                        and c2.args and expr_role(c2.args[0]) == "test":
                    out.append(_f(m, "CL-09", c2, "high", "The hyper-parameter search is fitted on test data.",
                                  "search-on-test"))
    # AIC/BIC order loops fitting ARIMA on the full series before the split
    for node in ast.walk(m.tree):
        if isinstance(node, ast.For):
            fits = [c for c in ast.walk(node) if isinstance(c, ast.Call)
                    and call_name(c) in {"ARIMA", "SARIMAX", "ExponentialSmoothing"} and c.args
                    and expr_role(c.args[0]) == "full"]
            uses_ic = any(isinstance(a, ast.Attribute) and a.attr in {"aic", "bic", "aicc"}
                          for a in ast.walk(node))
            if fits and uses_ic and (m.split_line is None or m.line_of(node) < m.split_line
                                     or m.split_line > m.line_of(node)):
                out.append(_f(m, "CL-09", node, "medium",
                              "The model order is chosen by information criterion with each candidate "
                              f"fitted on '{ast.unparse(fits[0].args[0])[:40]}', not on the training part only.",
                              "order-search-full-series"))
    return out


# ---------------------------------------------------------------------------
# CL-10  Outbreak threshold from the evaluation data
# ---------------------------------------------------------------------------

def detect_cl10(m: Module) -> list[Finding]:
    out: list[Finding] = []
    names = {call_name(c) for c in m.calls()}
    local_events = bool(names & CLS_METRICS) or bool(
        re.search(r"\b(f1|recall|precision|hit_rate|sensitivity|specificity|auc|pr_auc|roc_auc)\b", m.source, re.I))
    fits_here = any(call_name(c) in {"fit", "predict", "fit_predict", "train", "evaluate"} for c in m.calls())
    plotting_only = bool(re.search(r"(import matplotlib|import seaborn|import plotly|from matplotlib)", m.source)) \
        and not fits_here
    if not (local_events or (m.repo.get("event_metrics", False) and not plotting_only)):
        return out
    # p75_threshold = sorted(values)[int(0.75 * len(values))]
    for a in m.assigns:
        tname = a.targets[0].id if isinstance(a.targets[0], ast.Name) else ""
        v = a.value
        if THRESH_NAME.search(tname or "") and isinstance(v, ast.Subscript) and \
                any(isinstance(x, ast.Constant) and isinstance(x.value, float) and 0 < x.value < 1
                    for x in ast.walk(v.slice)) and expr_role(v) == "full":
            out.append(_f(m, "CL-10", v, "medium",
                          f"The threshold '{tname}' is a percentile of the whole series, evaluation period "
                          f"included.", "threshold-from-full-series"))
    for c in m.calls():
        nm = call_name(c)
        if nm not in {"quantile", "percentile", "nanpercentile", "nanquantile"}:
            continue
        if dotted(c.func).startswith(("np.", "numpy.")):
            data = c.args[0] if c.args else None
        else:
            data = c.func.value if isinstance(c.func, ast.Attribute) else None
        if data is None:
            continue
        stmt = m.enclosing(c, (ast.Assign,))
        tname = _assigned_column(stmt) if stmt else ""
        used_in_compare = isinstance(m.parent(c), ast.Compare) or (
            tname and any(isinstance(n, ast.Compare) and tname in names_in(n) for n in ast.walk(m.tree)))
        if not (THRESH_NAME.search(tname or "") or (used_in_compare and local_events)):
            continue
        role = expr_role(data)
        fn = m.enclosing(c, (ast.FunctionDef,))
        params = {a.arg for a in fn.args.args} if fn else set()
        data_names = names_in(data)
        if role == "test":
            out.append(_f(m, "CL-10", c, "high",
                          f"The threshold '{tname or nm}' is a {nm} of the test data.", "threshold-from-test"))
        elif fn is not None and data_names & params and \
                any(re.search(r"(y_true|true|actual|obs|label|target|y$)", p, re.I) for p in data_names & params):
            out.append(_f(m, "CL-10", c, "medium",
                          f"The threshold '{tname or nm}' is computed inside {fn.name}() from the labels being "
                          f"scored, so it moves with each evaluation set.", "threshold-from-scored-labels"))
        elif role == "full":
            # full-role data includes the evaluation period wherever the split happens
            out.append(_f(m, "CL-10", c, "medium",
                          f"The threshold '{tname or nm}' is a {nm} of the full series, evaluation period "
                          f"included.", "threshold-from-full-series"))
    return out


DETECTORS = {
    "CL-05": detect_cl05,
    "CL-06": detect_cl06,
    "CL-07": detect_cl07,
    "CL-09": detect_cl09,
    "CL-10": detect_cl10,
    "CL-11": detect_cl11,
}
