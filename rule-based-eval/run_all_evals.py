#!/usr/bin/env python3
"""Run evaluation across all eval-sets and their models.

Discovers eval-sets from input/ subfolders, matches each to the corresponding
GT SVG folder in eval-set/, evaluates every model inside, and saves a
consolidated JSON summary.

Usage:
    python run_all_evals.py                            # all eval-sets
    python run_all_evals.py arxiv-eval molmo-eval      # specific eval-sets
    python run_all_evals.py --models RL gemini-3-pro   # restrict models
    python run_all_evals.py --out results.json         # custom output path
    python run_all_evals.py --tables-only              # print tables from saved JSON, no eval
    python run_all_evals.py --tables-only --out x.json # load tables from specific JSON
    python run_all_evals.py --superset                 # also report scores over the shared
                                                       # clean-sample subset (models failing
                                                       # >51% are excluded with 0s)
"""

import sys, re, json, argparse
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "lib"))
sys.path.insert(0, str(BASE))

from main import evaluate_standalone
from utils import ZERO_SHAPE_SCORES, ZERO_ARROW_SCORES

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def find_gt_svg_dir(eval_set_name):
    """Find the GT SVG folder for a given eval-set name."""
    base_name = eval_set_name.removesuffix("-eval")
    candidates = [
        BASE / "eval-set" / eval_set_name,
        BASE / "eval-set" / base_name,
    ]
    for c in candidates:
        if c.exists() and list(c.glob("*.svg")):
            return c
    return None


def match_model_files(model_dir, gt_stems):
    """Map each GT stem to a model SVG file.

    Model files may have a trailing suffix after the GT stem
    (e.g. 'arxiv_svg__foo_qwen3vl.svg' matches GT 'arxiv_svg__foo.svg').
    Exact match is tried first, then prefix match.
    """
    all_model = {f.stem: f for f in model_dir.glob("*.svg")}
    matched = {}
    for sid in gt_stems:
        if sid in all_model:
            matched[sid] = all_model[sid]
            continue
        for stem, f in all_model.items():
            if stem.startswith(sid + "_") or stem.startswith(sid + "-"):
                matched[sid] = f
                break
    return matched


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def avg(per_sample):
    totals = defaultdict(list)
    for scores in per_sample.values():
        for k, v in scores.items():
            totals[k].append(v)
    return {k: sum(v) / len(v) for k, v in totals.items()} if totals else {}


def avg_over(per_sample, sids):
    """Average per_sample scores restricted to the given sample ids."""
    subset = {sid: per_sample[sid] for sid in sids if sid in per_sample}
    return avg(subset)


def evaluate_model(model_dir, gt_svg_dir, verbose=False):
    gt_samples = {f.stem: f for f in sorted(gt_svg_dir.glob("*.svg"))}
    matched    = match_model_files(model_dir, list(gt_samples))

    shape_ps, arrow_ps = {}, {}
    missing, errored = [], []
    clean = set()  # sids that were successfully evaluated (not missing, not errored)

    for sid, gt_svg in sorted(gt_samples.items()):
        gen_svg = matched.get(sid)
        if gen_svg is None:
            missing.append(sid)
            shape_ps[sid] = dict(ZERO_SHAPE_SCORES)
            arrow_ps[sid] = dict(ZERO_ARROW_SCORES)
            continue
        try:
            s_sc, _, a_sc, _ = evaluate_standalone(gen_svg, gt_svg)
            shape_ps[sid] = s_sc
            arrow_ps[sid] = a_sc
            clean.add(sid)
        except Exception as e:
            if verbose:
                print(f"      ERROR {sid}: {e}")
            errored.append(sid)
            shape_ps[sid] = dict(ZERO_SHAPE_SCORES)
            arrow_ps[sid] = dict(ZERO_ARROW_SCORES)

    s_avg = avg(shape_ps)
    a_avg = avg(arrow_ps)
    return s_avg, a_avg, missing, errored, shape_ps, arrow_ps, clean


# ---------------------------------------------------------------------------
# Superset computation
# ---------------------------------------------------------------------------

# Models with >SUPERSET_HIGH_FAIL_THRESH of samples missing or errored are
# excluded from the superset intersection and receive 0s.
SUPERSET_HIGH_FAIL_THRESH = 0.51


def compute_superset(model_data, n_total):
    """Compute superset scores for each model.

    model_data: dict of model_name -> {
        "shape_ps": per-sample shape scores (zeros for missing/errored),
        "arrow_ps": per-sample arrow scores (zeros for missing/errored),
        "clean":    set of successfully evaluated sids,
        "missing":  list, "errored": list
    }
    n_total: total number of GT samples

    High-failure models (>SUPERSET_HIGH_FAIL_THRESH missing+errored) are excluded
    from the superset intersection and receive 0s for all superset metrics.

    Returns:
        superset_sids: set of sample ids in the superset
        superset_scores: dict model_name -> (s_avg, a_avg)
        high_failure: set of model names flagged as high-failure
    """
    high_failure = {
        name for name, d in model_data.items()
        if (len(d["missing"]) + len(d["errored"])) > SUPERSET_HIGH_FAIL_THRESH * n_total
    }

    valid_names = [n for n in model_data if n not in high_failure]

    if not valid_names:
        superset_sids = set()
    else:
        # Intersection of clean samples across valid (non-high-failure) models only
        superset_sids = model_data[valid_names[0]]["clean"].copy()
        for name in valid_names[1:]:
            superset_sids &= model_data[name]["clean"]

    superset_scores = {}
    for name, d in model_data.items():
        if name in high_failure or not superset_sids:
            superset_scores[name] = (dict(ZERO_SHAPE_SCORES), dict(ZERO_ARROW_SCORES))
        else:
            s_avg = avg_over(d["shape_ps"], superset_sids)
            a_avg = avg_over(d["arrow_ps"], superset_sids)
            superset_scores[name] = (s_avg, a_avg)

    return superset_sids, superset_scores, high_failure


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

SHAPE_METRICS = ["label", "type", "fill_color", "fill_style", "stroke_color",
                 "border_style", "position", "font", "aspect_ratio", "composite"]
ARROW_METRICS = ["source", "dest", "head", "head_size", "curve", "color", "overlap", "composite"]

# Eval-sets and models excluded by default
EXCLUDED_EVAL_SETS = {"shapes-and-arrows-eval-all-models"}
EXCLUDED_MODELS    = {"inference", "RL_new"}


def print_eval_set_table(eval_set_name, model_results, superset_info=None):
    """model_results: list of (model_name, s_avg, a_avg, missing, errored)
    superset_info: optional (superset_sids, superset_scores, excluded) tuple
    """
    models = [r[0] for r in model_results]
    max_w  = max(len(m) for m in models)
    col_w  = 8

    print(f"\n{'='*90}")
    print(f"  {eval_set_name}")
    print(f"{'='*90}")

    for section, metrics in [("SHAPE", SHAPE_METRICS), ("ARROW", ARROW_METRICS)]:
        print(f"\n  -- {section} --")
        header = f"  {'Metric':<16}" + "".join(f"  {m[:8]:>{col_w}}" for m in models)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for metric in metrics:
            row = f"  {metric:<16}"
            for _, s_avg, a_avg, *_ in model_results:
                v = (s_avg if section == "SHAPE" else a_avg).get(metric, 0.0)
                row += f"  {v:>{col_w}.3f}"
            print(row)

    print(f"\n  {'Model':<{max_w}}  {'Shapes':>8}  {'Arrows':>8}  {'Overall':>8}  {'Missing':>7}  {'Errored':>7}")
    print("  " + "-" * (max_w + 50))
    rows = sorted(model_results, key=lambda r: -(r[1].get("composite",0) + r[2].get("composite",0)))
    for model_name, s_avg, a_avg, missing, errored in rows:
        sc = s_avg.get("composite", 0)
        ac = a_avg.get("composite", 0)
        print(f"  {model_name:<{max_w}}  {sc:>8.3f}  {ac:>8.3f}  {(sc+ac)/2:>8.3f}  {len(missing):>7}  {len(errored):>7}")

    if superset_info is not None:
        superset_sids, superset_scores, high_failure = superset_info
        n_sup = len(superset_sids)
        excl_str = (f"  (excluded: {', '.join(sorted(high_failure))})"
                    if high_failure else "")
        print(f"\n  -- SUPERSET ({n_sup} samples){excl_str} --")
        print(f"  {'Model':<{max_w}}  {'Shapes':>8}  {'Arrows':>8}  {'Overall':>8}  {'Note':>12}")
        print("  " + "-" * (max_w + 52))
        sup_rows = sorted(model_results, key=lambda r: -(
            superset_scores[r[0]][0].get("composite", 0) +
            superset_scores[r[0]][1].get("composite", 0)))
        for model_name, *_ in sup_rows:
            s_s, a_s = superset_scores[model_name]
            sc = s_s.get("composite", 0)
            ac = a_s.get("composite", 0)
            note = "excluded" if model_name in high_failure else ""
            print(f"  {model_name:<{max_w}}  {sc:>8.3f}  {ac:>8.3f}  {(sc+ac)/2:>8.3f}  {note:>12}")


def print_summary_tables(consolidated):
    """Print cross-eval-set summary: overall score and missing/errored per model."""
    eval_sets = list(consolidated.keys())
    all_models = sorted({m for es in consolidated.values() for m in es})

    col_w   = 14
    name_w  = max(len(m) for m in all_models)

    def header_row():
        return f"  {'Model':<{name_w}}" + "".join(f"  {e[:col_w]:>{col_w}}" for e in eval_sets)

    # Overall score table
    print(f"\n\n{'='*90}")
    print("SUMMARY — OVERALL SCORE  (50% shapes + 50% arrows)")
    print(f"{'='*90}")
    h = header_row()
    print(h)
    print("  " + "-" * (len(h) - 2))
    for model in all_models:
        row = f"  {model:<{name_w}}"
        for es in eval_sets:
            v = consolidated.get(es, {}).get(model, {}).get("overall", None)
            row += f"  {v:>{col_w}.3f}" if v is not None else f"  {'—':>{col_w}}"
        print(row)

    # Superset summary (if present in any eval-set)
    has_superset = any(
        any("superset_overall" in d for d in es_data.values())
        for es_data in consolidated.values()
    )
    if has_superset:
        print(f"\n\n{'='*90}")
        print("SUMMARY — SUPERSET OVERALL SCORE  (clean samples only; models with >51% miss/error excluded=0)")
        print(f"{'='*90}")
        print(h)
        print("  " + "-" * (len(h) - 2))
        # Superset N row
        n_row = f"  {'N (superset)':<{name_w}}"
        for es in eval_sets:
            es_data = consolidated.get(es, {})
            n_sup = None
            for d in es_data.values():
                n_sup = d.get("superset_n")
                if n_sup is not None:
                    break
            n_row += f"  {f'N={n_sup}' if n_sup is not None else '—':>{col_w}}"
        print(n_row)
        print("  " + "-" * (len(h) - 2))
        for model in all_models:
            row = f"  {model:<{name_w}}"
            for es in eval_sets:
                entry = consolidated.get(es, {}).get(model, {})
                v = entry.get("superset_overall", None)
                hf = entry.get("superset_high_failure", False)
                if v is None:
                    row += f"  {'—':>{col_w}}"
                elif hf:
                    row += f"  {'(excl)':>{col_w}}"
                else:
                    row += f"  {v:>{col_w}.3f}"
            print(row)

    # Missing + errored table
    print(f"\n\n{'='*90}")
    print("SUMMARY — MISSING / ERRORED SAMPLES")
    print(f"{'='*90}")
    print(h)
    print("  " + "-" * (len(h) - 2))
    for model in all_models:
        row = f"  {model:<{name_w}}"
        for es in eval_sets:
            entry = consolidated.get(es, {}).get(model)
            if entry is None:
                row += f"  {'—':>{col_w}}"
            else:
                miss = entry.get("missing_count", 0)
                err  = entry.get("errored_count", 0)
                row += f"  {f'{miss}m/{err}e':>{col_w}}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _setup():
    """Create the expected top-level directories and print usage instructions."""
    for d in [BASE / "eval-set", BASE / "input", BASE / "output"]:
        d.mkdir(exist_ok=True)
        print(f"  {d.relative_to(BASE)}/")

    print("""
Directory layout ready. Next steps:

  1. Place ground-truth SVGs in:
       eval-set/<name>/*.svg

  2. Place model inference SVGs in:
       input/<eval-set-name>/<model-name>/*.svg
     SVG filenames must match the GT stems (prefix match is also accepted).

  3. Run evaluation:
       python run_all_evals.py --superset
""")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all eval-sets and models.")
    parser.add_argument("eval_sets", nargs="*",
                        help="Eval-set names to run (default: all in input/)")
    parser.add_argument("--models",      nargs="*",       help="Restrict to specific model names")
    parser.add_argument("--out",         default="output/all_eval_results.json",
                        help="Output JSON path (default: output/all_eval_results.json)")
    parser.add_argument("--tables-only", action="store_true",
                        help="Skip evaluation; print tables from existing JSON")
    parser.add_argument("--superset",    action="store_true",
                        help="Also report scores over the shared clean-sample subset; "
                             "models failing >51%% of samples are excluded (given 0s)")
    parser.add_argument("--setup",       action="store_true",
                        help="Create the expected input/, eval-set/, and output/ directory layout")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-sample errors")
    args = parser.parse_args()

    if args.setup:
        _setup()
        return

    out_path = BASE / args.out

    # --tables-only: just load JSON and print
    if args.tables_only:
        if not out_path.exists():
            print(f"ERROR: {out_path} not found. Run without --tables-only first.")
            sys.exit(1)
        consolidated = json.loads(out_path.read_text(encoding="utf-8"))
        for eval_set_name, models_data in consolidated.items():
            model_results = []
            for model_name, data in models_data.items():
                s_avg   = data.get("shapes", {})
                a_avg   = data.get("arrows", {})
                missing = data.get("missing_samples", [])
                errored = [""] * data.get("errored_count", 0)
                model_results.append((model_name, s_avg, a_avg, missing, errored))
            print_eval_set_table(eval_set_name, model_results)
        print_summary_tables(consolidated)
        return

    input_root = BASE / "input"
    if not input_root.exists():
        print("ERROR: input/ not found"); sys.exit(1)

    # Discover eval-sets
    if args.eval_sets:
        eval_set_names = args.eval_sets
    else:
        eval_set_names = sorted(d.name for d in input_root.iterdir()
                                if d.is_dir() and d.name not in EXCLUDED_EVAL_SETS)

    print(f"Eval-sets : {', '.join(eval_set_names)}")
    if args.models:
        print(f"Models    : {', '.join(args.models)}")
    if args.superset:
        print(f"Superset  : enabled (exclude threshold: >{SUPERSET_HIGH_FAIL_THRESH*100:.0f}% missing/errored -> 0s)")
    print()

    consolidated = {}

    for eval_set_name in eval_set_names:
        eval_set_dir = input_root / eval_set_name
        if not eval_set_dir.exists():
            print(f"WARNING: input/{eval_set_name} not found, skipping"); continue

        gt_dir = find_gt_svg_dir(eval_set_name)
        if not gt_dir:
            print(f"WARNING: no GT SVGs found for '{eval_set_name}', skipping"); continue

        model_dirs = sorted(d for d in eval_set_dir.iterdir()
                            if d.is_dir() and d.name not in EXCLUDED_MODELS)
        if args.models:
            model_dirs = [d for d in model_dirs if d.name in args.models]
        if not model_dirs:
            print(f"  {eval_set_name}: no models found, skipping"); continue

        n_gt = len(list(gt_dir.glob("*.svg")))
        print(f"\n[{eval_set_name}]  GT: {gt_dir}  ({n_gt} samples)  Models: {', '.join(d.name for d in model_dirs)}")

        model_results = []
        eval_set_json = {}
        per_model_data = {}  # for superset computation

        for model_dir in model_dirs:
            print(f"  {model_dir.name}...", end="", flush=True)
            s_avg, a_avg, missing, errored, shape_ps, arrow_ps, clean = \
                evaluate_model(model_dir, gt_dir, verbose=args.verbose)
            sc = s_avg.get("composite", 0)
            ac = a_avg.get("composite", 0)
            print(f"  shapes={sc:.3f}  arrows={ac:.3f}  overall={(sc+ac)/2:.3f}"
                  f"  miss={len(missing)}  err={len(errored)}")
            model_results.append((model_dir.name, s_avg, a_avg, missing, errored))
            eval_set_json[model_dir.name] = {
                "overall":         (sc + ac) / 2,
                "shapes":          s_avg,
                "arrows":          a_avg,
                "missing_count":   len(missing),
                "errored_count":   len(errored),
                "missing_samples": missing,
            }
            per_model_data[model_dir.name] = {
                "shape_ps": shape_ps,
                "arrow_ps": arrow_ps,
                "clean":    clean,
                "missing":  missing,
                "errored":  errored,
            }

        # Superset computation
        superset_info = None
        if args.superset:
            superset_sids, superset_scores, high_failure = compute_superset(per_model_data, n_gt)
            superset_info = (superset_sids, superset_scores, high_failure)
            n_sup = len(superset_sids)
            print(f"  Superset: {n_sup}/{n_gt} samples"
                  + (f"  (excluded: {', '.join(sorted(high_failure))})"
                     if high_failure else ""))
            for model_name in per_model_data:
                s_s, a_s = superset_scores[model_name]
                sc_s = s_s.get("composite", 0)
                ac_s = a_s.get("composite", 0)
                is_hf = model_name in high_failure
                eval_set_json[model_name]["superset_overall"]      = (sc_s + ac_s) / 2
                eval_set_json[model_name]["superset_shapes"]        = s_s
                eval_set_json[model_name]["superset_arrows"]        = a_s
                eval_set_json[model_name]["superset_n"]             = n_sup
                eval_set_json[model_name]["superset_high_failure"]  = is_hf

        print_eval_set_table(eval_set_name, model_results, superset_info=superset_info)
        consolidated[eval_set_name] = eval_set_json

    print_summary_tables(consolidated)

    # Save JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
