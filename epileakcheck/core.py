"""Core model and shared analysis for EpiLeakCheck.

A `Module` wraps one parsed source file (a .py script or the code cells of a
.ipynb notebook) and pre-computes the facts every detector needs:

* the AST with parent links,
* inferred target (label) column names,
* names that hold train / validation / test data,
* the first line at which the data is split,
* simple assignment records (name -> value nodes, in source order).

Detectors are pure functions ``detect(module) -> list[Finding]``.
Everything here is static: no user code is imported or executed.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1}


@dataclass
class Finding:
    rule: str            # candidate ID, e.g. "CL-07"
    name: str            # short name of the candidate
    file: str
    line: int
    confidence: str      # high | medium | low
    message: str         # what was found, in plain words
    snippet: str = ""    # the source line
    pattern: str = ""    # which sub-pattern of the rule fired
    kn_type: str = ""    # Kapoor & Narayanan leakage type, if any
    fix: str = ""        # short remediation hint
    end_line: int = 0    # last line of the flagged statement
    segment: list = field(default_factory=list)   # [{"line", "code", "flagged"}] with context
    related: list = field(default_factory=list)   # [{"line", "label", "code"}] supporting lines
    what: str = ""       # the problem in one sentence
    why: str = ""        # why it distorts the reported performance
    how: list = field(default_factory=list)       # ordered fix steps
    example_before: str = ""
    example_after: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_MAGIC = re.compile(r"^\s*[!%]")


def notebook_source(path: Path) -> str:
    """Concatenate code cells of a notebook; IPython magics become comments."""
    nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    cells = nb.get("cells") or nb.get("worksheets", [{}])[0].get("cells", [])
    out: list[str] = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        lines = ["# " + ln if _MAGIC.match(ln) else ln for ln in src.splitlines()]
        out.append(f"# --- cell {i} ---")
        out.extend(lines)
    return "\n".join(out) + "\n"


def read_source(path: Path) -> str:
    if path.suffix == ".ipynb":
        return notebook_source(path)
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def dotted(node: ast.AST) -> str:
    """Best-effort dotted name: np.quantile, self.scaler.fit, df.groupby."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return dotted(node.func)
    if isinstance(node, ast.Subscript):
        return dotted(node.value)
    return ""


def call_name(node: ast.Call) -> str:
    """Final attribute or function name of a call: df.rolling(3) -> 'rolling'."""
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def method_chain(node: ast.AST) -> tuple[ast.AST, list[ast.Call]]:
    """Unwind a method chain.

    ``df['c'].shift(1).rolling(3).mean()`` -> (df['c'], [shift(1), rolling(3), mean()])
    The list is ordered from innermost (first applied) to outermost.
    Subscripts in the middle of a chain (groupby('u')['c']) are walked through.
    """
    calls: list[ast.Call] = []
    cur = node
    while True:
        if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
            calls.append(cur)
            cur = cur.func.value
        elif isinstance(cur, ast.Subscript) and isinstance(cur.value, ast.Call) \
                and isinstance(cur.value.func, ast.Attribute):
            # groupby(...)['col'] : keep walking, remember the column via subscript
            calls.append(cur)  # type: ignore[arg-type]
            cur = cur.value
        else:
            break
    calls.reverse()
    return cur, calls


def str_const(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def subscript_key(node: ast.AST) -> str | None:
    """df['cases'] -> 'cases'; df.loc[:, 'cases'] -> 'cases'."""
    if isinstance(node, ast.Subscript):
        sl = node.slice
        s = str_const(sl)
        if s is not None:
            return s
        if isinstance(sl, ast.JoinedStr):
            return ast.unparse(sl)  # f"target_{h}" -> keeps the literal parts for name checks
        if isinstance(sl, ast.Tuple) and sl.elts:
            return str_const(sl.elts[-1])
    if isinstance(node, ast.Call) and call_name(node) == "col" and node.args:  # polars pl.col("y")
        return str_const(node.args[0])
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        # df.cases (only for lower-case attribute names that look like columns)
        return None
    return None


def columns_referenced(node: ast.AST) -> set[str]:
    """All string keys used in subscripts anywhere inside node."""
    cols: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and call_name(sub) == "col":
            k = subscript_key(sub)
            if k:
                cols.add(k)
        if isinstance(sub, ast.Subscript):
            k = subscript_key(sub)
            if k is not None:
                cols.add(k)
            # groupby('u')[['a','b']]
            if isinstance(sub.slice, (ast.List, ast.Tuple)):
                for e in sub.slice.elts:
                    s = str_const(e)
                    if s:
                        cols.add(s)
    return cols


def names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def const_number(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = const_number(node.operand)
        return -v if v is not None else None
    return None


def first_arg(call: ast.Call, kw: str | None = None) -> ast.AST | None:
    if call.args:
        return call.args[0]
    if kw:
        for k in call.keywords:
            if k.arg == kw:
                return k.value
    return None


def kwarg(call: ast.Call, name: str) -> ast.AST | None:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def shift_sign(call: ast.Call) -> int:
    """+1 for shift(k>0) or shift() (default 1), -1 for shift(-k) or shift(1-h), 0 unknown."""
    arg = first_arg(call, "periods")
    if arg is None:
        return 1
    v = const_number(arg)
    if v is not None:
        return 1 if v > 0 else (-1 if v < 0 else 0)
    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
        return -1
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Sub):
        # shift(1 - h) / shift(-(h-1)) patterns
        lv = const_number(arg.left)
        if lv is not None and lv <= 1:
            return -1
    return 1  # shift(lag) with a variable: assume a positive lag


# ---------------------------------------------------------------------------
# Name roles
# ---------------------------------------------------------------------------

TEST_RE = re.compile(r"(^|_)(te|tst|ho|holdout|hold_out|oos)(_|$|\d)|(?<!la)test", re.I)
VAL_RE = re.compile(r"(^|_)(val|valid|validation|dev)(_|$|\d)", re.I)
TRAIN_RE = re.compile(r"train|(^|_)(tr|fit)(_|$)", re.I)

TARGET_NAME_RE = re.compile(r"^(target|label|y_col|ycol|target_col|label_col|target_column|"
                            r"y_name|target_name|outcome|response|TARGET|LABEL|Y_COL|TARGET_COL)$",
                            re.I)
COMMON_TARGETS = {"cases", "total_cases", "casos", "dengue_cases", "new_cases", "confirmed",
                  "incidence", "count", "counts", "value", "y", "Patients", "patients",
                  "admissions", "hospitalizations", "deaths", "case_count", "num_cases",
                  "weekly_cases", "monthly_cases", "Dengue_Cases", "cases_count"}

WEATHER_RE = re.compile(r"(temp|tavg|tmin|tmax|tair|rain|precip|prcp|humid|rh_|_rh|dew|"
                        r"wind|wspd|ndvi|evi|climate|weather|enso|oni|soi|sst|vpd|"
                        r"pressure|cloud|sun|solar|larva|larval|ovitrap|mobility|search|trends)",
                        re.I)
LAG_WORD_RE = re.compile(r"(lag|shift|prev|past|_l\d|_t\d|minus|_m\d|ago|before|rolling_lag)", re.I)

METRIC_FUNCS = {"mean_absolute_error", "mean_squared_error", "r2_score", "mean_absolute_percentage_error",
                "root_mean_squared_error", "f1_score", "accuracy_score", "roc_auc_score",
                "precision_score", "recall_score", "mae", "mse", "rmse", "mape", "smape",
                "average_precision_score", "wis", "crps", "log_loss", "explained_variance_score",
                "median_absolute_error", "balanced_accuracy_score"}

FIT_CLASSES = re.compile(r"(Scaler|Normalizer|Imputer|Transformer|Encoder|PCA|Discretizer|"
                         r"Binarizer|KBinsDiscretizer|TruncatedSVD|StandardScaler)$")


def role_of(name: str) -> str:
    """'test' | 'val' | 'train' | 'full' for a variable name."""
    if TEST_RE.search(name):
        return "test"
    if VAL_RE.search(name):
        return "val"
    if TRAIN_RE.search(name):
        return "train"
    return "full"


def expr_role(node: ast.AST) -> str:
    """Role of the data an expression reads: the most 'evaluation-like' name in it."""
    roles = {role_of(n) for n in names_in(node)}
    # string keys count too: stack["y_ho_raw"], splits["test"]
    roles |= {role_of(s.value) for s in ast.walk(node) if isinstance(s, ast.Constant)
              and isinstance(s.value, str) and len(s.value) < 40 and re.fullmatch(r"[A-Za-z_][\w]*", s.value)}
    for r in ("test", "val", "train"):
        if r in roles:
            return r
    return "full"


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

@dataclass
class Assign:
    targets: list[ast.AST]
    value: ast.AST
    line: int


@dataclass
class Module:
    path: str
    source: str
    tree: ast.Module
    lines: list[str] = field(default_factory=list)
    parents: dict[int, ast.AST] = field(default_factory=dict)
    assigns: list[Assign] = field(default_factory=list)
    targets: set[str] = field(default_factory=set)
    split_line: int | None = None
    split_kinds: set[str] = field(default_factory=set)
    target_vars: set[str] = field(default_factory=set)   # names bound to a target column string
    target_line: int = 0                                   # where the target is defined
    root: str = ""                                         # scan root, for locating data files
    repo: dict = field(default_factory=dict)               # repository-level facts set by scan_path

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_source(cls, source: str, path: str = "<string>", root: str = "") -> "Module":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # exported notebooks often keep shell/magic lines; comment them out and retry
            source = "\n".join("# " + ln if _MAGIC.match(ln) else ln for ln in source.splitlines()) + "\n"
            tree = ast.parse(source)
        m = cls(path=path, source=source, tree=tree, lines=source.splitlines(), root=root)
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                m.parents[id(child)] = node
        m._collect_assigns()
        m._infer_targets()
        m._find_split()
        return m

    @classmethod
    def from_path(cls, path: Path, root: str = "") -> "Module":
        return cls.from_source(read_source(path), str(path), root or str(path.parent))

    # ---- helpers ----------------------------------------------------------
    def line_of(self, node: ast.AST) -> int:
        return getattr(node, "lineno", 0) or 0

    def snippet(self, node: ast.AST) -> str:
        ln = self.line_of(node)
        if 0 < ln <= len(self.lines):
            return self.lines[ln - 1].strip()[:200]
        return ""

    def parent(self, node: ast.AST) -> ast.AST | None:
        return self.parents.get(id(node))

    def enclosing_assign(self, node: ast.AST) -> ast.Assign | ast.AugAssign | ast.AnnAssign | None:
        cur = self.parent(node)
        while cur is not None and not isinstance(cur, (ast.stmt,)):
            cur = self.parent(cur)
        if isinstance(cur, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            return cur
        return None

    def enclosing(self, node: ast.AST, kinds: tuple) -> ast.AST | None:
        cur = self.parent(node)
        while cur is not None:
            if isinstance(cur, kinds):
                return cur
            cur = self.parent(cur)
        return None

    def calls(self) -> Iterator[ast.Call]:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call):
                yield n

    def span_of(self, node: ast.AST) -> tuple[int, int]:
        """Line span of the whole statement that contains node (multi-line calls included)."""
        stmt = node if isinstance(node, ast.stmt) else self.enclosing(node, (ast.stmt,))
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With, ast.FunctionDef, ast.Try)):
            # for compound statements flag the header, not the whole body
            start = stmt.lineno
            end = (stmt.body[0].lineno - 1) if getattr(stmt, "body", None) else stmt.lineno
            return start, max(start, end)
        tgt = stmt or node
        start = getattr(tgt, "lineno", 0) or self.line_of(node)
        end = getattr(tgt, "end_lineno", None) or start
        return start, end

    def segment(self, start: int, end: int, context: int = 2) -> list[dict]:
        lo, hi = max(1, start - context), min(len(self.lines), end + context)
        return [{"line": i, "code": self.lines[i - 1], "flagged": start <= i <= end} for i in range(lo, hi + 1)]

    def line_code(self, ln: int) -> str:
        return self.lines[ln - 1].strip()[:200] if 0 < ln <= len(self.lines) else ""

    def finding(self, rule: str, name: str, node: ast.AST, confidence: str, message: str,
                pattern: str = "", kn: str = "", fix: str = "") -> Finding:
        start, end = self.span_of(node)
        line = self.line_of(node) or start
        return Finding(rule=rule, name=name, file=self.path, line=line,
                       confidence=confidence, message=message, snippet=self.snippet(node),
                       pattern=pattern, kn_type=kn, fix=fix, end_line=end,
                       segment=self.segment(min(start, line), end))

    # ---- pre-analysis ------------------------------------------------------
    def _collect_assigns(self) -> None:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Assign):
                self.assigns.append(Assign(list(n.targets), n.value, n.lineno))
            elif isinstance(n, ast.AnnAssign) and n.value is not None:
                self.assigns.append(Assign([n.target], n.value, n.lineno))
        self.assigns.sort(key=lambda a: a.line)

    def _infer_targets(self) -> None:
        t: set[str] = set()
        for a in self.assigns:
            for tgt in a.targets:
                tname = tgt.id if isinstance(tgt, ast.Name) else ""
                # TARGET = "cases"
                s = str_const(a.value)
                if s and TARGET_NAME_RE.match(tname or ""):
                    t.add(s)
                # y = df["cases"]  /  y_train = train["cases"]
                if tname and re.match(r"^y(_?(train|test|val|valid|true|all|full))?$", tname, re.I):
                    k = subscript_key(a.value)
                    if k:
                        t.add(k)
                    if isinstance(a.value, ast.Call):
                        base, chain = method_chain(a.value)
                        k = subscript_key(base)
                        if k:
                            t.add(k)
                # X, y = df.drop(columns=[T]), df[T]
                if isinstance(tgt, ast.Tuple) and isinstance(a.value, ast.Tuple):
                    for lt, rv in zip(tgt.elts, a.value.elts):
                        if isinstance(lt, ast.Name) and re.match(r"^y", lt.id):
                            k = subscript_key(rv)
                            if k:
                                t.add(k)
        # keyword arguments: target="cases", label_col="y"
        for c in self.calls():
            for k in c.keywords:
                if k.arg and TARGET_NAME_RE.match(k.arg):
                    s = str_const(k.value)
                    if s:
                        t.add(s)
        # function defaults: def f(df, target="cases")
        for fn in ast.walk(self.tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = fn.args.args[-len(fn.args.defaults):] if fn.args.defaults else []
                for a, d in zip(args, fn.args.defaults):
                    if TARGET_NAME_RE.match(a.arg) and str_const(d):
                        t.add(str_const(d))  # type: ignore[arg-type]
        # fall back to common outcome names that are actually used as columns
        if not t:
            used = columns_referenced(self.tree)
            t |= used & COMMON_TARGETS
        self.targets = {x for x in t if x}
        self.target_line = 0
        for a in self.assigns:
            names = [tg.id for tg in a.targets if isinstance(tg, ast.Name)]
            if any(re.match(r"^(y|y_?train|y_?all|y_?full|target|label|TARGET|LABEL)$", n) for n in names) or \
                    (any(str_const(n_) in self.targets for n_ in ast.walk(a.value) if isinstance(n_, ast.Constant))
                     and any(re.match(r"^y", n, re.I) for n in names)):
                self.target_line = a.line
                break
        for a in self.assigns:
            s = str_const(a.value)
            if s in self.targets:
                for tg in a.targets:
                    if isinstance(tg, ast.Name):
                        self.target_vars.add(tg.id)

    def _find_split(self) -> None:
        lines: list[int] = []
        for c in self.calls():
            nm = call_name(c)
            if nm in {"train_test_split", "TimeSeriesSplit", "KFold", "split", "temporal_train_test_split",
                      "GroupKFold", "StratifiedKFold", "ShuffleSplit", "ExpandingWindowSplitter",
                      "SlidingWindowSplitter", "split_before", "split_after"}:
                lines.append(c.lineno)
                self.split_kinds.add(nm)
        for a in self.assigns:
            for tgt in a.targets:
                names = [n.id for n in ast.walk(tgt) if isinstance(n, ast.Name)]
                if any(role_of(n) in ("train", "test", "val") for n in names):
                    if isinstance(a.value, (ast.Subscript, ast.Call, ast.Tuple)):
                        lines.append(a.line)
                        self.split_kinds.add("assign")
                        break
        self.split_line = min(lines) if lines else None

    # convenience predicates used by several detectors -------------------
    def refs_target(self, node: ast.AST) -> bool:
        return bool(columns_referenced(node) & self.targets) or bool(names_in(node) & self.target_vars) \
            or any(str_const(n) in self.targets for n in ast.walk(node) if isinstance(n, ast.Constant))

    def before_split(self, node: ast.AST) -> bool:
        return self.split_line is not None and self.line_of(node) < self.split_line


TEST_FILE_RE = re.compile(r"(^test_.*\.py$|_test\.py$|^conftest\.py$)")


def iter_python_files(root: Path, include_notebooks: bool = True,
                      include_tests: bool = False) -> Iterable[Path]:
    skip = {".git", "venv", ".venv", "env", "site-packages", "node_modules", "__pycache__", "build", "dist"}
    if not include_tests:
        skip |= {"tests", "test", "testing"}
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if any(part in skip for part in p.relative_to(root).parts):
            continue
        if not include_tests and TEST_FILE_RE.search(p.name):
            continue
        if p.suffix == ".py" or (include_notebooks and p.suffix == ".ipynb"):
            yield p
