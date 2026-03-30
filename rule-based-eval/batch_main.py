#!/usr/bin/env python3
"""Batch evaluation across all model folders in input/.

Usage:
    python batch_main.py                        # evaluate all models in input/
    python batch_main.py --models RL RL_new     # restrict to specific models
    python batch_main.py --gt-svg path/to/svgs  # explicit GT SVG folder
    python batch_main.py -v                     # verbose (show per-sample errors)

Features:
  - Auto-discovers all model folders in input/
  - Reports per-model sample coverage
  - Evaluates every GT sample; models missing a sample receive score 0
  - Tracks per-sample per-metric scores
  - Writes output/<model_name>/{sid}_shapes.txt, {sid}_arrows.txt, annotation SVGs
  - Saves full JSON summary to output/batch_results.json
"""

import sys, re, argparse, json, shutil
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "lib"))
sys.path.insert(0, str(BASE))   # so we can import from main.py

from main import evaluate_standalone
from utils import ZERO_SHAPE_SCORES, ZERO_ARROW_SCORES

# ---------------------------------------------------------------------------
# Behaviour flag — flip to revert to strict intersection mode
# ---------------------------------------------------------------------------

# When True  (default): evaluate every GT sample; models missing a sample
#   receive 0.0 for all metrics on that sample.
# When False (legacy):  only evaluate samples present in ALL models.
USE_ZERO_FOR_MISSING = True

# ---------------------------------------------------------------------------
# Paths — detect eval-set location automatically
# ---------------------------------------------------------------------------


def _find_gt_svg_dir():
    """Find the GT SVG folder (flat folder of *.svg files)."""
    for c in [BASE / "eval-set"]:
        if c.exists() and list(c.glob("*.svg")):
            return c
    return None


GT_SVG_DIR = _find_gt_svg_dir()

INPUT_DIR  = BASE / "input"
OUTPUT_DIR = BASE / "output"

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

SHAPE_METRICS = [
    "label", "type", "fill_color", "fill_style", "stroke_color",
    "border_style", "position", "font", "aspect_ratio",
]
ARROW_METRICS = ["source", "dest", "head", "head_size", "curve", "color", "overlap"]


# ---------------------------------------------------------------------------
# Sample list helpers
# ---------------------------------------------------------------------------

def discover_models():
    if not INPUT_DIR.exists():
        print(f"ERROR: {INPUT_DIR} does not exist"); sys.exit(1)
    return sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])


def all_gt_samples():
    """Sorted list of every sample ID present in the GT SVG folder."""
    if not GT_SVG_DIR:
        return []
    return sorted(f.stem for f in GT_SVG_DIR.glob("*.svg"))


def compute_sample_list(model_dirs):
    """Return (sample_list, per-model coverage dict).

    USE_ZERO_FOR_MISSING=True  -> sample_list = all GT samples
    USE_ZERO_FOR_MISSING=False -> sample_list = intersection across all models + GT
    """
    model_samples = {d.name: set(f.stem for f in d.glob("*.svg")) for d in model_dirs}
    gt_all = set(all_gt_samples())

    if USE_ZERO_FOR_MISSING:
        sample_list = sorted(gt_all)
    else:
        common = gt_all
        for ids in model_samples.values():
            common = common & ids
        sample_list = sorted(common)

    coverage = {}
    for name, ids in model_samples.items():
        have    = ids & gt_all
        missing = gt_all - ids
        coverage[name] = {"have": len(have), "missing": len(missing),
                          "missing_ids": sorted(missing)}
    return sample_list, coverage


def report_coverage(model_dirs, sample_list, coverage):
    mode = "ZERO-FOR-MISSING" if USE_ZERO_FOR_MISSING else "INTERSECTION"
    print(f"\n{'='*70}")
    print(f"MODEL COVERAGE REPORT  [{mode}]")
    print(f"{'='*70}")
    if GT_SVG_DIR:
        print(f"Eval-set : {GT_SVG_DIR}")
    else:
        print("WARNING: eval-set not found")
    print(f"GT samples to evaluate: {len(sample_list)}")
    print()
    print(f"  {'Model':<50}  {'Have':>5}  {'Missing':>7}")
    print(f"  {'-'*65}")
    for d in model_dirs:
        info = coverage[d.name]
        flag = "  <<" if info["missing"] > 0 else ""
        print(f"  {d.name:<50}  {info['have']:>5}  {info['missing']:>7}{flag}")
    if not USE_ZERO_FOR_MISSING:
        print(f"\n  Intersection size: {len(sample_list)}")
    print()

# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def copy_svg(src, sample_id, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / f"{sample_id}.svg").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def evaluate_model(model_dir, sample_list, gt_svg_dir, verbose=False):
    """Evaluate one model across sample_list using standalone (GT-SVG-only) evaluation.

    Missing samples get ZERO_SHAPE_SCORES / ZERO_ARROW_SCORES scores.
    Annotation SVGs and text reports are written to OUTPUT_DIR/<model_name>/.

    Returns:
        model_name        str
        shape_per_sample  dict[sid -> metric_dict]
        arrow_per_sample  dict[sid -> metric_dict]
        missing           list[sid]   samples the model had no SVG for
        errored           list[sid]   samples that raised exceptions
    """
    model_name  = model_dir.name
    ann_dir     = OUTPUT_DIR / model_name
    results_dir = ann_dir

    if ann_dir.exists():
        shutil.rmtree(ann_dir)
    ann_dir.mkdir(parents=True)

    shape_per_sample = {}
    arrow_per_sample = {}
    missing = []
    errored = []

    for sid in sample_list:
        src = model_dir / f"{sid}.svg"
        gt  = gt_svg_dir / f"{sid}.svg"

        if not src.exists():
            missing.append(sid)
            shape_per_sample[sid] = dict(ZERO_SHAPE_SCORES)
            arrow_per_sample[sid] = dict(ZERO_ARROW_SCORES)
            continue

        if not gt.exists():
            errored.append(sid)
            continue

        try:
            s_sc, s_rep, a_sc, a_rep = evaluate_standalone(
                src, gt, ann_dir=ann_dir, label=sid)
            (results_dir / f"{sid}_shapes.txt").write_text(s_rep, encoding="utf-8")
            (results_dir / f"{sid}_arrows.txt").write_text(a_rep, encoding="utf-8")
            shape_per_sample[sid] = s_sc
            arrow_per_sample[sid] = a_sc
        except Exception as e:
            if verbose:
                print(f"  [{model_name}] {sid} ERROR: {e}")
            errored.append(sid)
            shape_per_sample[sid] = dict(ZERO_SHAPE_SCORES)
            arrow_per_sample[sid] = dict(ZERO_ARROW_SCORES)

    return model_name, shape_per_sample, arrow_per_sample, missing, errored


def aggregate(per_sample):
    """Average per-sample scores -> {metric: mean}."""
    totals = defaultdict(list)
    for scores in per_sample.values():
        for k, v in scores.items():
            totals[k].append(v)
    return {k: sum(v) / len(v) for k, v in totals.items()}

# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def print_model_table(all_results, sample_count):
    models   = [r[0] for r in all_results]
    max_name = max(len(m) for m in models)
    mode_tag = "(mean, 0 for missing)" if USE_ZERO_FOR_MISSING else "(intersection mean)"

    def section(title, metrics, key):
        print(f"\n{'='*95}")
        print(f"{title}  {mode_tag}")
        print(f"{'='*95}")
        header = f"{'Model':<{max_name}}"
        for m in metrics:
            header += f"  {m[:9]:>9}"
        print(header)
        print("-" * len(header))
        for model_name, s_avg, a_avg, missing, errored in all_results:
            data = s_avg if key == "shapes" else a_avg
            row  = f"{model_name:<{max_name}}"
            for m in metrics:
                row += f"  {data.get(m, 0.0):>9.3f}"
            row += f"  miss={len(missing)}"
            print(row)

    section("SHAPE METRICS", SHAPE_METRICS, "shapes")
    section("ARROW METRICS", ARROW_METRICS, "arrows")

    print(f"\n{'='*95}")
    print(f"OVERALL  (50% shapes + 50% arrows)  [{sample_count} GT samples]")
    print(f"{'='*95}")
    print(f"{'Model':<{max_name}}  {'Shapes':>8}  {'Arrows':>8}  {'Overall':>8}  Missing  Errored")
    print("-" * (max_name + 50))
    rows = []
    for model_name, s_avg, a_avg, missing, errored in all_results:
        sc = s_avg.get("composite", 0.0)
        ac = a_avg.get("composite", 0.0)
        rows.append((model_name, sc, ac, (sc + ac) / 2, len(missing), len(errored)))
    for model_name, sc, ac, ov, nm, ne in sorted(rows, key=lambda x: -x[3]):
        print(f"{model_name:<{max_name}}  {sc:>8.3f}  {ac:>8.3f}  {ov:>8.3f}  {nm:>7}  {ne:>7}")

# ---------------------------------------------------------------------------
# Metric variance chart
# ---------------------------------------------------------------------------

def print_metric_variance(all_results):
    """Print per-metric variance across models (higher = models disagree more)."""
    if len(all_results) < 2:
        return

    def var(metric, key):
        vals = [s_avg.get(metric, 0.0) if key == "shapes" else a_avg.get(metric, 0.0)
                for _, s_avg, a_avg, *_ in all_results]
        mean = sum(vals) / len(vals) if vals else 0.0
        return sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0

    print(f"\n{'='*70}")
    print("METRIC VARIANCE  (higher = models differ more on this metric)")
    print(f"{'='*70}")
    print("\n  Shape:")
    for v, m in sorted(((var(m, "shapes"), m) for m in SHAPE_METRICS), reverse=True):
        print(f"    {m:<15}  {v:.4f}  {'#' * int(v * 300)}")
    print("\n  Arrow:")
    for v, m in sorted(((var(m, "arrows"), m) for m in ARROW_METRICS), reverse=True):
        print(f"    {m:<15}  {v:.4f}  {'#' * int(v * 300)}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate all models in input/")
    parser.add_argument("-v", "--verbose",   action="store_true", help="Show per-sample errors")
    parser.add_argument("--models",          nargs="*",           help="Restrict to these model folder names")
    parser.add_argument("--gt-svg",          default=None,        help="GT SVG folder (auto-detected if omitted)")
    args = parser.parse_args()

    global GT_SVG_DIR
    if args.gt_svg:
        GT_SVG_DIR = Path(args.gt_svg)

    # Eval-set check
    if not GT_SVG_DIR:
        print("ERROR: GT SVG folder not found. Expected: eval-set/ (flat folder of *.svg files)")
        print("Use --gt-svg to specify the folder explicitly.")
        sys.exit(1)
    n_svg = len(list(GT_SVG_DIR.glob("*.svg")))
    print(f"Eval-set : {GT_SVG_DIR}  ({n_svg} GT SVGs)")
    print(f"Mode     : {'ZERO-FOR-MISSING (all GT samples)' if USE_ZERO_FOR_MISSING else 'INTERSECTION'}")

    # Discover models
    model_dirs = discover_models()
    if args.models:
        model_dirs = [d for d in model_dirs if d.name in args.models]
    if not model_dirs:
        print("No model directories found in input/"); sys.exit(1)

    # Sample list + coverage report
    sample_list, coverage = compute_sample_list(model_dirs)
    report_coverage(model_dirs, sample_list, coverage)

    if not sample_list:
        print("ERROR: No samples to evaluate."); sys.exit(1)

    print(f"Evaluating {len(sample_list)} samples across {len(model_dirs)} models ...")
    print(f"Annotation output: output/<model_name>/\n")

    # Run evaluation
    all_results = []   # (model_name, s_avg, a_avg, missing, errored)

    for model_dir in model_dirs:
        print(f"  [{model_dir.name}]", end="", flush=True)
        model_name, shape_ps, arrow_ps, missing, errored = evaluate_model(
            model_dir, sample_list, GT_SVG_DIR, verbose=args.verbose)
        s_avg = aggregate(shape_ps)
        a_avg = aggregate(arrow_ps)
        sc = s_avg.get("composite", 0.0)
        ac = a_avg.get("composite", 0.0)
        print(f"  shapes={sc:.3f}  arrows={ac:.3f}  overall={(sc+ac)/2:.3f}"
              f"  miss={len(missing)}  err={len(errored)}")
        all_results.append((model_name, s_avg, a_avg, missing, errored))

    # Summary table
    print_model_table(all_results, len(sample_list))

    # Metric variance chart
    print_metric_variance(all_results)

    # JSON summary
    summary = {}
    for model_name, s_avg, a_avg, missing, errored in all_results:
        sc = s_avg.get("composite", 0.0); ac = a_avg.get("composite", 0.0)
        summary[model_name] = {
            "overall": (sc + ac) / 2,
            "shapes":  s_avg,
            "arrows":  a_avg,
            "missing_count": len(missing),
            "errored_count": len(errored),
            "missing_samples": missing,
        }
    out = OUTPUT_DIR / "batch_results.json"
    OUTPUT_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll outputs: {OUTPUT_DIR}/")
    print(f"  batch_results.json + <model_name>/{{sid}}_shapes.txt, {{sid}}_arrows.txt, {{sid}}_shapes.svg, {{sid}}_arrows.svg")


if __name__ == "__main__":
    main()
