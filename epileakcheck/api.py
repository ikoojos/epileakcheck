"""Public API.

    from epileakcheck import scan_path, scan_source
    report = scan_path("path/to/repo")
    for f in report.findings: print(f.rule, f.file, f.line, f.message)
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .core import CONFIDENCE_ORDER, Finding, Module, iter_python_files
from .detectors import DETECTORS as CL_DETECTORS, RULES as CL_RULES, _windowed
from .fm_detectors import (FM_DETECTORS, FM_RULES, TIME_SIGNAL, TRANSFER_CLAIM, VINTAGE_GUARD)

__version__ = "0.3.2"

DETECTORS = {**FM_DETECTORS, **CL_DETECTORS}
RULES = {**FM_RULES, **CL_RULES}


@dataclass
class Report:
    root: str
    files_scanned: int = 0
    findings: list[Finding] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def by_rule(self) -> dict[str, list[Finding]]:
        out: dict[str, list[Finding]] = {}
        for f in self.findings:
            out.setdefault(f.rule, []).append(f)
        return out

    def rules_hit(self, min_confidence: str = "low") -> set[str]:
        lo = CONFIDENCE_ORDER[min_confidence]
        return {f.rule for f in self.findings if CONFIDENCE_ORDER[f.confidence] >= lo}

    def to_dict(self) -> dict:
        return {
            "tool": "EpiLeakCheck", "version": __version__, "root": self.root,
            "files_scanned": self.files_scanned,
            "summary": {r: len(v) for r, v in sorted(self.by_rule().items())},
            "findings": [f.to_dict() for f in self.findings],
            "errors": self.errors,
        }


def _select(rules: Iterable[str] | None) -> dict:
    if not rules:
        return DETECTORS
    wanted = {r.upper() for r in rules}
    unknown = wanted - set(DETECTORS)
    if unknown:
        raise ValueError(f"unknown rule(s): {sorted(unknown)}; available: {sorted(DETECTORS)}")
    return {k: v for k, v in DETECTORS.items() if k in wanted}


def _repo_facts(modules: list[Module], docs: str = "") -> dict:
    """Facts that hold for the whole repository and change how per-file rules read."""
    src = "\n".join(m.source for m in modules)
    return {
        "temporal": bool(TIME_SIGNAL.search(src)),
        "vintage_guard": bool(VINTAGE_GUARD.search(src)),
        "transfer_claim": bool(TRANSFER_CLAIM.search(docs)),
        "windowed": any(_windowed(m) for m in modules),
        "event_metrics": bool(re.search(r"(f1_score|recall_score|precision_score|roc_auc_score|\bf1\b|\brecall\b|"
                                        r"pr_auc|roc_auc|hit_rate|sensitivity)", src, re.I)),
        "evaluates": bool(re.search(r"(mean_absolute_error|mean_squared_error|r2_score|\bmae\b|\brmse\b|"
                                    r"train_test_split|backtest)", src, re.I)),
    }


def _related_lines(f: Finding, m: Module, kinds: list[str]) -> list[dict]:
    """Supporting lines that, together with the flagged segment, make the failure."""
    import ast
    from .core import call_name, expr_role
    from .detectors import _is_metric_call
    out: list[dict] = []

    def add(line: int, label: str) -> None:
        if line and not (f.line <= line <= max(f.end_line, f.line)) and all(r["line"] != line for r in out):
            out.append({"line": line, "label": label, "code": m.line_code(line)})

    for k in kinds:
        if k == "split" and m.split_line:
            add(m.split_line, "data are split into training and evaluation parts here")
        elif k == "target" and m.target_line:
            add(m.target_line, "the target (label) is defined here")
        elif k == "time":
            for c in m.calls():
                if call_name(c) in {"to_datetime", "shift", "date_range", "isocalendar", "sort_values"}:
                    add(c.lineno, "rows are time-indexed here (dates, lags or time order)")
                    break
        elif k == "metric":
            for c in m.calls():
                if _is_metric_call(c) and any(expr_role(a) == "test" for a in c.args):
                    add(c.lineno, "the test-set score used for the choice is computed here")
                    break
        elif k == "feature":
            names = re.findall(r"'([A-Za-z_][\w]*)'", f.message)[:1]   # the feature the message names
            for name in names:
                hit = False
                for n in ast.walk(m.tree):
                    if isinstance(n, (ast.List, ast.Tuple)) and any(
                            isinstance(e, ast.Constant) and e.value == name for e in n.elts):
                        add(n.lineno, f"'{name}' is listed as a model input here")
                        hit = True
                        break
                if not hit:  # implicit selection: every column except the target
                    for n in ast.walk(m.tree):
                        if (isinstance(n, ast.ListComp) and "columns" in ast.unparse(n.generators[0].iter)) or \
                                (isinstance(n, ast.Call) and call_name(n) == "drop" and m.refs_target(n)):
                            add(n.lineno, f"inputs are all columns except the target, so '{name}' is included")
                            break
    return sorted(out, key=lambda r: r["line"])


def _enrich(f: Finding, m: Module) -> Finding:
    from .explain import explain
    info = explain(f.pattern) or {}
    f.what = info.get("what", f.message)
    f.why = info.get("why", "")
    f.how = list(info.get("how", [f.fix] if f.fix else []))
    f.example_before = info.get("before", "")
    f.example_after = info.get("after", "")
    f.related = _related_lines(f, m, info.get("related", []))
    return f


def _postprocess(findings: list[Finding], facts: dict) -> list[Finding]:
    """Repository-level adjustments. Same-period covariates in a repo that builds sliding windows
    (inputs strictly before the target) are only a weak signal."""
    out = []
    for f in findings:
        if f.rule == "CL-05" and f.pattern.startswith("same-period-covariates") and facts.get("windowed"):
            f.confidence = "low"
            f.message += " (The repository also builds sliding windows, which may lag these inputs.)"
        out.append(f)
    return out


def scan_module(module: Module, rules: Iterable[str] | None = None,
                min_confidence: str = "low") -> list[Finding]:
    lo = CONFIDENCE_ORDER[min_confidence]
    out: list[Finding] = []
    for rule, fn in _select(rules).items():
        out.extend(fn(module))
    out = _postprocess(out, module.repo)
    # one finding per rule and line: keep the most confident sub-pattern, mention the others
    best: dict = {}
    for f in out:
        k = (f.file, f.rule, f.line)
        if k not in best:
            best[k] = f
        else:
            keep, other = (f, best[k]) if CONFIDENCE_ORDER[f.confidence] > CONFIDENCE_ORDER[best[k].confidence] \
                else (best[k], f)
            if other.pattern != keep.pattern:
                keep.message += f" Also: {other.message}"
            best[k] = keep
    out = list(best.values())
    out = [_enrich(f, module) for f in out if CONFIDENCE_ORDER[f.confidence] >= lo]
    out.sort(key=lambda f: (f.file, f.line, f.rule))
    return out


def scan_source(source: str, filename: str = "<string>", rules: Iterable[str] | None = None,
                min_confidence: str = "low") -> Report:
    rep = Report(root=filename, files_scanned=1)
    try:
        mod = Module.from_source(source, filename)
        mod.repo = _repo_facts([mod])
        rep.findings = scan_module(mod, rules, min_confidence)
    except SyntaxError as e:
        rep.errors.append({"file": filename, "error": f"SyntaxError: {e}"})
    return rep


def _docs_text(root: Path) -> str:
    if root.is_file():
        root = root.parent
    txt = []
    for p in list(root.glob("README*"))[:3] + list(root.glob("docs/*.md"))[:10]:
        try:
            txt.append(p.read_text(encoding="utf-8", errors="replace")[:200_000])
        except OSError:
            pass
    return "\n".join(txt)


def scan_path(path: str | Path, rules: Iterable[str] | None = None, min_confidence: str = "low",
              include_notebooks: bool = True, include_tests: bool = False) -> Report:
    root = Path(path)
    rep = Report(root=str(root))
    base = str(root if root.is_dir() else root.parent)
    modules: list[Module] = []
    for p in iter_python_files(root, include_notebooks, include_tests):
        rep.files_scanned += 1
        try:
            modules.append(Module.from_path(p, base))
        except SyntaxError as e:
            rep.errors.append({"file": str(p), "error": f"SyntaxError: {e.msg} (line {e.lineno})"})
        except Exception as e:  # unreadable notebook etc.
            rep.errors.append({"file": str(p), "error": f"{type(e).__name__}: {e}"})
    facts = _repo_facts(modules, _docs_text(root))
    for mod in modules:
        mod.repo = facts
        try:
            rep.findings.extend(scan_module(mod, rules, min_confidence))
        except Exception as e:  # a detector bug must not stop the scan
            rep.errors.append({"file": mod.path, "error": "detector failure: " + "".join(
                traceback.format_exception_only(type(e), e)).strip()})
    rep.findings.sort(key=lambda f: (f.file, f.line, f.rule))
    return rep


def list_rules() -> list[dict]:
    return [{"rule": k, "name": n, "kn_type": kn, "fix": fix} for k, (n, kn, fix) in RULES.items()]
