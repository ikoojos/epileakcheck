"""Command line.

    epileakcheck scan PATH [--format text|markdown|json|sarif] [--brief] [--rules FM1,CL-07]
    epileakcheck explain FM1 | CL-07 | fit-before-split
    epileakcheck rules
    epileakcheck serve [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import sys

from .api import RULES, __version__, list_rules, scan_path
from .core import CONFIDENCE_ORDER
from .explain import E as EXPLAIN


def _sarif(report) -> dict:
    rules = [{"id": r["rule"], "name": r["name"], "help": {"text": r["fix"]}} for r in list_rules()]
    level = {"high": "error", "medium": "warning", "low": "note"}
    results = []
    for f in report.findings:
        text = f"{f.what or f.message} {f.why}".strip()
        if f.how:
            text += " Fix: " + " ".join(f"({i}) {s}" for i, s in enumerate(f.how, 1))
        loc = {"physicalLocation": {"artifactLocation": {"uri": f.file},
                                    "region": {"startLine": max(f.line, 1),
                                               "endLine": max(f.end_line, f.line, 1)}}}
        rel = [{"id": i, "message": {"text": r["label"]},
                "physicalLocation": {"artifactLocation": {"uri": f.file}, "region": {"startLine": r["line"]}}}
               for i, r in enumerate(f.related)]
        results.append({"ruleId": f.rule, "level": level[f.confidence], "message": {"text": text},
                        "locations": [loc], "relatedLocations": rel,
                        "properties": {"pattern": f.pattern, "confidence": f.confidence, "kn_type": f.kn_type}})
    return {"version": "2.1.0", "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [{"tool": {"driver": {"name": "EpiLeakCheck", "version": __version__, "rules": rules}},
                      "results": results}]}


def _lines_range(f) -> str:
    return f"line {f.line}" if f.end_line <= f.line else f"lines {f.line}-{f.end_line}"


def _text(report, brief: bool = False) -> str:
    out = [f"EpiLeakCheck {__version__}: {report.files_scanned} file(s) scanned, "
           f"{len(report.findings)} finding(s)", ""]
    for n, f in enumerate(report.findings, 1):
        out.append(f"[{n}] {f.rule} {f.name}  [{f.confidence}]")
        out.append(f"    {f.file}, {_lines_range(f)}")
        if brief:
            out.append(f"    {f.message}")
            out.append(f"    Fix: {f.fix}")
            out.append("")
            continue
        width = len(str(max([s['line'] for s in f.segment] + [r['line'] for r in f.related] + [1])))
        for s in f.segment:
            mark = ">>" if s["flagged"] else "  "
            out.append(f"    {mark} {str(s['line']).rjust(width)} | {s['code']}")
        for r in f.related:
            out.append(f"    -> {str(r['line']).rjust(width)} | {r['code']}    <- {r['label']}")
        out.append("")
        out.append(f"    What:  {f.what}")
        out.append(f"    Found: {f.message}")
        if f.why:
            out.append(f"    Why it matters: {f.why}")
        if f.how:
            out.append("    How to fix:")
            out += [f"      {i}. {s}" for i, s in enumerate(f.how, 1)]
        if f.example_before:
            out.append("    Example, before:")
            out += [f"      {ln}" for ln in f.example_before.splitlines()]
            out.append("    after:")
            out += [f"      {ln}" for ln in f.example_after.splitlines()]
        if f.kn_type:
            out.append(f"    Kapoor & Narayanan leakage type: {f.kn_type}")
        out.append("")
    for e in report.errors:
        out.append(f"skipped {e['file']}: {e['error']}")
    return "\n".join(out)


def _markdown(report) -> str:
    out = [f"# EpiLeakCheck report", "",
           f"{report.files_scanned} file(s) scanned in `{report.root}`, {len(report.findings)} finding(s).", ""]
    if report.findings:
        out += ["| # | Rule | Confidence | File | Lines |", "| --- | --- | --- | --- | --- |"]
        for n, f in enumerate(report.findings, 1):
            out.append(f"| {n} | {f.rule} {f.name} | {f.confidence} | `{f.file}` | {_lines_range(f).split(' ', 1)[1]} |")
        out.append("")
    for n, f in enumerate(report.findings, 1):
        out += [f"## {n}. {f.rule} {f.name} ({f.confidence})", "",
                f"`{f.file}`, {_lines_range(f)}", "", "```python"]
        out += [f"{'>>' if s['flagged'] else '  '} {s['line']:>4} | {s['code']}" for s in f.segment]
        out += ["```", ""]
        if f.related:
            out.append("Related lines:")
            out += [f"- line {r['line']}: `{r['code']}` ({r['label']})" for r in f.related]
            out.append("")
        out += [f"**What:** {f.what}", "", f"**Found:** {f.message}", ""]
        if f.why:
            out += [f"**Why it matters:** {f.why}", ""]
        if f.how:
            out.append("**How to fix:**")
            out += [f"{i}. {s}" for i, s in enumerate(f.how, 1)]
            out.append("")
        if f.example_before:
            out += ["Before:", "```python", f.example_before, "```", "After:", "```python", f.example_after, "```", ""]
        if f.kn_type:
            out += [f"Kapoor & Narayanan leakage type: {f.kn_type}", ""]
    if report.errors:
        out += ["## Skipped files", ""] + [f"- `{e['file']}`: {e['error']}" for e in report.errors]
    return "\n".join(out)


def _explain(key: str) -> str:
    key_up = key.upper()
    if key_up in RULES:
        name, kn, fix = RULES[key_up]
        pats = sorted(p for p in EXPLAIN if _pattern_rule(p) == key_up)
        lines = [f"{key_up}  {name}" + (f"  (K&N {kn})" if kn else ""), f"Fix in short: {fix}", "",
                 "Sub-patterns:"] + [f"  {p}: {EXPLAIN[p]['what']}" for p in pats]
        lines += ["", f"Run 'epileakcheck explain <sub-pattern>' for details."]
        return "\n".join(lines)
    info = EXPLAIN.get(key)
    if not info:
        return f"Unknown rule or pattern '{key}'. Try 'epileakcheck rules'."
    lines = [key, "", f"What: {info['what']}", f"Why it matters: {info['why']}", "How to fix:"]
    lines += [f"  {i}. {s}" for i, s in enumerate(info["how"], 1)]
    lines += ["Before:"] + [f"  {ln}" for ln in info["before"].splitlines()]
    lines += ["After:"] + [f"  {ln}" for ln in info["after"].splitlines()]
    return "\n".join(lines)


_PATTERN_RULE = {
    "FM1": ["random-train-test-split", "non-temporal-cv", "default-kfold", "integer-cv",
            "reversed-before-positional-split"],
    "FM2": ["notification-anchor-no-correction", "notification-anchor-onset-available",
            "latest-vintage-revisable-source", "provisional-file"],
    "FM3": ["spatial-feature-from-all-units", "random-split-transfer-claim"],
    "FM4": ["multi-year-load-unharmonised", "concat-year-files"],
    "CL-05": ["negative-shift-covariate", "same-period-covariates", "same-period-covariates-from-data",
              "lag-search-includes-zero"],
    "CL-06": ["rolling-target-starts-at-t"],
    "CL-07": ["fit-on-test", "fit-on-validation", "fit-before-split", "eval-normalised-with-own-stats",
              "normalisation-stat-full", "fillna-full-stat", "impute-full-array-stats", "backward-fill",
              "interpolate-both"],
    "CL-09": ["early-stop-on-test", "rank-test-scores", "select-by-test-metric", "keep-best-on-test",
              "order-search-full-series", "search-on-test"],
    "CL-10": ["threshold-from-full-series", "threshold-from-test", "threshold-from-scored-labels"],
    "CL-11": ["unshifted-rolling-on-target", "target-arithmetic"],
}


def _pattern_rule(p: str) -> str:
    for r, ps in _PATTERN_RULE.items():
        if p in ps:
            return r
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="epileakcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan a file or directory")
    s.add_argument("path")
    s.add_argument("--format", choices=["text", "markdown", "json", "sarif"], default="text")
    s.add_argument("--brief", action="store_true", help="one-line message and fix per finding (text format)")
    s.add_argument("--rules", default="", help="comma-separated rule IDs (default: all)")
    s.add_argument("--min-confidence", choices=list(CONFIDENCE_ORDER), default="low")
    s.add_argument("--no-notebooks", action="store_true")
    s.add_argument("--include-tests", action="store_true", help="also scan tests/ and test_*.py")
    s.add_argument("--fail-on", choices=["never", "low", "medium", "high"], default="never",
                   help="exit 1 if any finding at or above this confidence (for CI)")
    e = sub.add_parser("explain", help="explain a rule (FM1, CL-07) or a sub-pattern (fit-before-split)")
    e.add_argument("key")
    sub.add_parser("rules", help="list rules")
    sv = sub.add_parser("serve", help="run the HTTP API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)

    if a.cmd == "rules":
        for r in list_rules():
            print(f"{r['rule']}  {r['name']}" + (f"  (K&N {r['kn_type']})" if r["kn_type"] else ""))
        return 0
    if a.cmd == "explain":
        print(_explain(a.key))
        return 0
    if a.cmd == "serve":
        from .server import serve
        serve(a.host, a.port)
        return 0
    rules = [r.strip() for r in a.rules.split(",") if r.strip()] or None
    rep = scan_path(a.path, rules=rules, min_confidence=a.min_confidence,
                    include_notebooks=not a.no_notebooks, include_tests=a.include_tests)
    if a.format == "json":
        print(json.dumps(rep.to_dict(), indent=2))
    elif a.format == "sarif":
        print(json.dumps(_sarif(rep), indent=2))
    elif a.format == "markdown":
        print(_markdown(rep))
    else:
        print(_text(rep, brief=a.brief))
    if a.fail_on != "never" and rep.rules_hit(a.fail_on):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
