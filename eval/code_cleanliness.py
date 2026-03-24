#!/usr/bin/env python3
"""
code_cleanliness.py

Evaluates the "Semantic Cleanliness" metric for every SVG model folder under
`rendered_svg/`.  Statistics are broken down per dataset sub-subfolder
(e.g. arxiv_p2f, molmo, starvector) and then aggregated per model.

Metric definition (paper-friendly):
  Clean = (B + K) / (N + eps)

  where:
    B = rect + circle + ellipse   (basic shapes)
    K = line + polyline            (connectors)
    C = path + polygon             (complex shapes)
    N = B + K + C                  (total shapes, excluding text)
    eps = 1e-9                     (avoid division by zero)

  Interpretation:
    Clean → 1.0  :  mostly simple/structural shapes  (diagram-like)
    Clean → 0.0  :  mostly paths/polygons             (artistic/illustrative)

Results are saved to:
  benchmark_data/code_cleanliness_results.md
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
RENDERED_SVG_ROOT = Path(
    "xxx"
)
OUTPUT_MD = Path(
    "xxx.md"
)

EPS = 1e-9


# ─────────────────────────────────────────────
# Core metric helpers  (mirrors diagram_score.py)
# ─────────────────────────────────────────────

def count_svg_primitives(svg_content: str) -> dict:
    """Count SVG element tags by fast substring matching."""
    s = svg_content.lower()
    return {
        "rect":     s.count("<rect"),
        "circle":   s.count("<circle"),
        "ellipse":  s.count("<ellipse"),
        "line":     s.count("<line"),
        "polyline": s.count("<polyline"),
        "path":     s.count("<path"),
        "polygon":  s.count("<polygon"),
        "text":     svg_content.count("<text"),   # original case – safer
    }


def compute_cleanliness(svg_content: str) -> Optional[float]:
    """
    Return the Semantic Cleanliness score, or None if the SVG has no elements.

    Clean = (B + K) / (N + eps)
    """
    c = count_svg_primitives(svg_content)
    B = c["rect"] + c["circle"] + c["ellipse"]
    K = c["line"] + c["polyline"]
    C = c["path"] + c["polygon"]
    N = B + K + C
    T = c["text"]

    if N == 0 and T == 0:
        return None

    return (B + K) / (N + EPS)


# ─────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────

def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": float("nan"), "p10": float("nan"),
                "p50": float("nan"), "p90": float("nan"),
                "min": float("nan"), "max": float("nan")}
    xs = sorted(values)
    n = len(xs)

    def q(p: float) -> float:
        if n == 1:
            return xs[0]
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        w = idx - lo
        return xs[lo] * (1 - w) + xs[hi] * w

    return {
        "count": n,
        "mean":  sum(xs) / n,
        "p10":   q(0.10),
        "p50":   q(0.50),
        "p90":   q(0.90),
        "min":   xs[0],
        "max":   xs[-1],
    }


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def score_svg_files(svg_files: List[Path]) -> Tuple[List[float], int, int]:
    """
    Read and score a list of SVG paths.
    Returns (scores, failed_read, empty_skipped).
    """
    scores: List[float] = []
    failed_read = 0
    empty_skipped = 0
    for fp in svg_files:
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            failed_read += 1
            continue
        score = compute_cleanliness(content)
        if score is None:
            empty_skipped += 1
            continue
        scores.append(score)
    return scores, failed_read, empty_skipped


def evaluate_model_folder(model_dir: Path) -> dict:
    """
    For a top-level model folder, compute per-sub-subfolder stats and an aggregate.

    If the model folder has immediate sub-subfolders containing SVGs, report each
    sub-subfolder separately.  If it has no sub-subfolders (SVGs live directly
    inside model_dir), treat the whole folder as a single "dataset".
    """
    # Collect immediate subdirectories that actually contain SVGs
    subdirs = sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and any(d.glob("*.svg"))
    )

    # Fallback: SVGs directly in model_dir (no dataset sub-subfolders)
    if not subdirs:
        direct_svgs = sorted(model_dir.glob("*.svg"))
        scores, failed, empty = score_svg_files(direct_svgs)
        sub_results = [{
            "dataset":       model_dir.name,
            "total_svgs":    len(direct_svgs),
            "analyzed":      len(scores),
            "failed_read":   failed,
            "empty_skipped": empty,
            "stats":         summarize(scores),
            "scores":        scores,
        }]
    else:
        sub_results = []
        for sub in subdirs:
            svgs = sorted(sub.glob("*.svg"))
            scores, failed, empty = score_svg_files(svgs)
            sub_results.append({
                "dataset":       sub.name,
                "total_svgs":    len(svgs),
                "analyzed":      len(scores),
                "failed_read":   failed,
                "empty_skipped": empty,
                "stats":         summarize(scores),
                "scores":        scores,
            })

    # Aggregate across all sub-subfolders
    all_scores: List[float] = []
    for r in sub_results:
        all_scores.extend(r["scores"])

    return {
        "model":       model_dir.name,
        "sub_results": sub_results,
        "aggregate":   summarize(all_scores),
        "total_svgs":  sum(r["total_svgs"] for r in sub_results),
        "analyzed":    sum(r["analyzed"]   for r in sub_results),
    }


# ─────────────────────────────────────────────
# Markdown helpers
# ─────────────────────────────────────────────

def fmt(v: float, decimals: int = 4) -> str:
    if math.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


def build_markdown(model_results: List[dict]) -> str:
    lines: List[str] = []

    lines += [
        "# SVG Semantic Cleanliness Report",
        "",
        "**Metric**: `Clean = (B + K) / (N + ε)`",
        "",
        "| Symbol | Meaning |",
        "|--------|---------|",
        "| B | `<rect>` + `<circle>` + `<ellipse>` (basic shapes) |",
        "| K | `<line>` + `<polyline>` (connectors) |",
        "| C | `<path>` + `<polygon>` (complex shapes) |",
        "| N | B + K + C (total shapes) |",
        "| ε | 1e-9 (numerical stability) |",
        "",
        "> **Interpretation**: Clean → 1.0 = diagram-like (simple shapes dominant);  "
        "Clean → 0.0 = artistic/illustrative (path-heavy).",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "One row per model (aggregate across all its dataset sub-subfolders).",
        "",
        "| Model | SVGs | Analyzed | Mean | p10 | p50 | p90 | Min | Max |",
        "|-------|-----:|--------:|-----:|----:|----:|----:|----:|----:|",
    ]

    for m in model_results:
        s = m["aggregate"]
        lines.append(
            f"| {m['model']} "
            f"| {m['total_svgs']} "
            f"| {m['analyzed']} "
            f"| {fmt(s['mean'])} "
            f"| {fmt(s['p10'])} "
            f"| {fmt(s['p50'])} "
            f"| {fmt(s['p90'])} "
            f"| {fmt(s['min'])} "
            f"| {fmt(s['max'])} |"
        )

    # Grand overall
    all_scores_flat: List[float] = []
    for m in model_results:
        for r in m["sub_results"]:
            all_scores_flat.extend(r["scores"])
    overall = summarize(all_scores_flat)
    total_svgs = sum(m["total_svgs"] for m in model_results)
    total_analyzed = sum(m["analyzed"] for m in model_results)
    lines.append(
        f"| **OVERALL** "
        f"| {total_svgs} "
        f"| {total_analyzed} "
        f"| **{fmt(overall['mean'])}** "
        f"| {fmt(overall['p10'])} "
        f"| {fmt(overall['p50'])} "
        f"| {fmt(overall['p90'])} "
        f"| {fmt(overall['min'])} "
        f"| {fmt(overall['max'])} |"
    )

    lines += [
        "",
        "---",
        "",
        "## Per-Model Breakdown (by Dataset Sub-subfolder)",
        "",
    ]

    for m in model_results:
        ag = m["aggregate"]
        lines += [
            f"### {m['model']}",
            "",
            f"- **Total SVGs**: {m['total_svgs']}  |  **Analyzed**: {m['analyzed']}",
            "",
            "| Dataset | SVGs | Analyzed | Failed | Empty | Mean | p10 | p50 | p90 | Min | Max |",
            "|---------|-----:|--------:|-------:|------:|-----:|----:|----:|----:|----:|----:|",
        ]
        for r in m["sub_results"]:
            s = r["stats"]
            lines.append(
                f"| {r['dataset']} "
                f"| {r['total_svgs']} "
                f"| {r['analyzed']} "
                f"| {r['failed_read']} "
                f"| {r['empty_skipped']} "
                f"| {fmt(s['mean'])} "
                f"| {fmt(s['p10'])} "
                f"| {fmt(s['p50'])} "
                f"| {fmt(s['p90'])} "
                f"| {fmt(s['min'])} "
                f"| {fmt(s['max'])} |"
            )
        # Aggregate row for this model
        lines.append(
            f"| **Aggregate** "
            f"| {m['total_svgs']} "
            f"| {m['analyzed']} "
            f"| — | — "
            f"| **{fmt(ag['mean'])}** "
            f"| {fmt(ag['p10'])} "
            f"| {fmt(ag['p50'])} "
            f"| {fmt(ag['p90'])} "
            f"| {fmt(ag['min'])} "
            f"| {fmt(ag['max'])} |"
        )
        lines.append("")

    lines += [
        "---",
        "",
        "## Overall Aggregate (all models and datasets combined)",
        "",
        f"- **Total SVGs analyzed**: {overall['count']}",
        "",
        "| Statistic    | Value |",
        "|--------------|-------|",
        f"| Mean         | {fmt(overall['mean'])} |",
        f"| p10          | {fmt(overall['p10'])} |",
        f"| p50 (median) | {fmt(overall['p50'])} |",
        f"| p90          | {fmt(overall['p90'])} |",
        f"| Min          | {fmt(overall['min'])} |",
        f"| Max          | {fmt(overall['max'])} |",
        "",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # All immediate subdirectories of RENDERED_SVG_ROOT
    model_dirs = sorted(
        d for d in RENDERED_SVG_ROOT.iterdir() if d.is_dir()
    )

    if not model_dirs:
        print("No model folders found. Exiting.")
        return

    print(f"Found {len(model_dirs)} model folder(s) under {RENDERED_SVG_ROOT.name}/")
    print()

    model_results: List[dict] = []

    for model_dir in model_dirs:
        print(f"[{model_dir.name}]", flush=True)
        result = evaluate_model_folder(model_dir)
        model_results.append(result)
        ag = result["aggregate"]
        for r in result["sub_results"]:
            s = r["stats"]
            print(
                f"  {r['dataset']:50s}  svgs={r['total_svgs']:4d}  "
                f"analyzed={r['analyzed']:4d}  mean={fmt(s['mean'])}  p50={fmt(s['p50'])}"
            )
        print(
            f"  {'[aggregate]':50s}  svgs={result['total_svgs']:4d}  "
            f"analyzed={result['analyzed']:4d}  mean={fmt(ag['mean'])}  p50={fmt(ag['p50'])}"
        )
        print()

    md_content = build_markdown(model_results)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(md_content, encoding="utf-8")
    print(f"✅ Report written to: {OUTPUT_MD.resolve()}")


if __name__ == "__main__":
    main()
