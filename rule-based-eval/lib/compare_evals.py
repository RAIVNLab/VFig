#!/usr/bin/env python3
"""Compare JSON-backed vs SVG-reconstruction evaluation scores across models.

JSON-backed  : uses ground-truth metadata from eval-set/objects/*.json  (authoritative)
Standalone   : reconstructs metadata from the GT SVG on the fly (main.py method)

The delta (JSON - standalone) shows where SVG reconstruction diverges from ground truth.

Usage:
    python lib/compare_evals.py                                   # default eval-set, all models
    python lib/compare_evals.py RL RL_new                         # specific models
    python lib/compare_evals.py --eval-set shapes-and-arrows      # specific eval-set
    python lib/compare_evals.py --eval-set shapes-and-arrows RL   # eval-set + specific models
    python lib/compare_evals.py --highlights                       # generate 5 peculiar sample highlights
"""

import sys, re, shutil, argparse
from pathlib import Path
from collections import defaultdict
from xml.etree import ElementTree as ET

LIB_DIR = Path(__file__).resolve().parent
BASE    = LIB_DIR.parent

sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(BASE))

from utils import strip_ns, ZERO_SHAPE_SCORES, ZERO_ARROW_SCORES
from eval_shapes import evaluate_shapes_file
from eval_arrows import evaluate_arrows_file
from main import evaluate_standalone, reconstruct_meta

SHAPE_METRICS = ["label", "type", "fill_color", "fill_style", "stroke_color",
                 "border_style", "position", "font", "aspect_ratio", "composite"]
ARROW_METRICS = ["source", "dest", "head", "head_size", "curve", "color", "overlap", "composite"]

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _find_gt_svg_dir(eval_set=None):
    """Find GT SVG directory, optionally scoped to a named eval-set."""
    if eval_set:
        base_name = eval_set.removesuffix("-eval")
        candidates = [
            BASE / "eval-set" / eval_set,
            BASE / "eval-set" / base_name,
        ]
    else:
        candidates = [
            BASE / "eval-set",
        ]
    for c in candidates:
        if c.exists() and list(c.glob("*.svg")):
            return c
    return None


def _find_objects_dir(eval_set=None):
    """Find objects JSON directory, optionally scoped to a named eval-set."""
    if eval_set:
        base_name = eval_set.removesuffix("-eval")
        candidates = [
            BASE / "eval-set" / eval_set / "objects",
            BASE / "eval-set" / base_name / "objects",
        ]
    else:
        candidates = [BASE / "eval-set" / "objects"]
    for c in candidates:
        if c.exists() and list(c.glob("*.json")):
            return c
    return None


def _find_input_dir(eval_set=None):
    """Find model input directory for a given eval-set."""
    if eval_set:
        # Try exact match first, then with -eval suffix
        for name in [eval_set, eval_set + "-eval",
                     eval_set.removesuffix("-eval"),
                     eval_set.removesuffix("-eval") + "-eval"]:
            d = BASE / "input" / name
            if d.exists():
                return d
    return BASE / "input"

# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def _avg(per_sample):
    totals = defaultdict(list)
    for scores in per_sample.values():
        for k, v in scores.items():
            totals[k].append(v)
    return {k: sum(v) / len(v) for k, v in totals.items()} if totals else {}


def evaluate_json(model_dir, gt_svg_dir, objects_dir):
    """JSON-backed evaluation using ground-truth metadata files."""
    import tempfile
    shape_ps, arrow_ps = {}, {}
    gt_samples = {f.stem: f for f in sorted(gt_svg_dir.glob("*.svg"))}

    for sid, gt_svg in gt_samples.items():
        json_file = objects_dir / f"{sid}.json"
        gen_svg   = model_dir / f"{sid}.svg"

        if not json_file.exists() or not gen_svg.exists():
            shape_ps[sid] = dict(ZERO_SHAPE_SCORES)
            arrow_ps[sid] = dict(ZERO_ARROW_SCORES)
            continue
        try:
            s_sc, _ = evaluate_shapes_file(gen_svg, gt_svg, json_file, None, sid)
            a_sc, _ = evaluate_arrows_file(gen_svg, gt_svg, json_file, None, sid)
            shape_ps[sid] = s_sc
            arrow_ps[sid] = a_sc
        except Exception:
            shape_ps[sid] = dict(ZERO_SHAPE_SCORES)
            arrow_ps[sid] = dict(ZERO_ARROW_SCORES)

    return _avg(shape_ps), _avg(arrow_ps)


def evaluate_standalone_model(model_dir, gt_svg_dir):
    """SVG-reconstruction evaluation (main.py method)."""
    shape_ps, arrow_ps = {}, {}
    gt_samples = {f.stem: f for f in sorted(gt_svg_dir.glob("*.svg"))}

    for sid, gt_svg in gt_samples.items():
        gen_svg = model_dir / f"{sid}.svg"
        if not gen_svg.exists():
            shape_ps[sid] = dict(ZERO_SHAPE_SCORES)
            arrow_ps[sid] = dict(ZERO_ARROW_SCORES)
            continue
        try:
            s_sc, _, a_sc, _ = evaluate_standalone(gen_svg, gt_svg)
            shape_ps[sid] = s_sc
            arrow_ps[sid] = a_sc
        except Exception:
            shape_ps[sid] = dict(ZERO_SHAPE_SCORES)
            arrow_ps[sid] = dict(ZERO_ARROW_SCORES)

    return _avg(shape_ps), _avg(arrow_ps)

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _composite(s_avg, a_avg):
    return (s_avg.get("composite", 0) + a_avg.get("composite", 0)) / 2


def print_table(models, j_results, s_results):
    """One column per model showing JSON - standalone gap per metric."""
    col_w = 8
    header = f"{'Metric':<18}" + "".join(f"  {m[:7]:>{col_w}}" for m in models)
    print(header)
    print("-" * len(header))

    for section, metrics in [("SHAPE", SHAPE_METRICS), ("ARROW", ARROW_METRICS)]:
        print(f"\n  -- {section} --")
        for metric in metrics:
            row = f"  {metric:<16}"
            for m in models:
                js_avg, ja_avg = j_results[m]
                ss_avg, sa_avg = s_results[m]
                jv = (js_avg if section == "SHAPE" else ja_avg).get(metric, 0.0)
                sv = (ss_avg if section == "SHAPE" else sa_avg).get(metric, 0.0)
                row += f"  {jv - sv:+{col_w}.3f}"
            print(row)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare JSON-backed vs standalone eval.")
    parser.add_argument("models", nargs="*", help="Model names to evaluate (default: all)")
    parser.add_argument("--eval-set", default=None,
                        help="Eval-set name (e.g. shapes-and-arrows, arxiv-eval). "
                             "Determines GT SVG dir, objects dir, and input model dir.")
    args = parser.parse_args()

    gt_svg_dir  = _find_gt_svg_dir(args.eval_set)
    objects_dir = _find_objects_dir(args.eval_set)
    input_dir   = _find_input_dir(args.eval_set)

    if not gt_svg_dir:
        print("ERROR: GT SVG folder not found."); return
    if not objects_dir:
        print("ERROR: objects/ not found — JSON-backed eval unavailable.")
        print("       Run standalone-only with batch_main.py instead.")
        return
    if not input_dir or not input_dir.exists():
        print(f"ERROR: input directory not found ({input_dir})."); return

    if args.models:
        models = args.models
    else:
        models = sorted(d.name for d in input_dir.iterdir() if d.is_dir())

    if not models:
        print(f"No models found in {input_dir}"); return

    print(f"Eval-set : {args.eval_set or '(default)'}")
    print(f"GT SVGs  : {gt_svg_dir}")
    print(f"Objects  : {objects_dir}")
    print(f"Input    : {input_dir}")
    print(f"Models   : {', '.join(models)}\n")

    j_results = {}   # model -> (s_avg, a_avg) JSON-backed
    s_results = {}   # model -> (s_avg, a_avg) standalone

    for model in models:
        model_dir = input_dir / model
        if not model_dir.exists():
            print(f"  {model}: not found in {input_dir}, skipping"); continue

        print(f"  {model}...", end="", flush=True)
        js, ja = evaluate_json(model_dir, gt_svg_dir, objects_dir)
        ss, sa = evaluate_standalone_model(model_dir, gt_svg_dir)
        j_results[model] = (js, ja)
        s_results[model] = (ss, sa)
        jc = _composite(js, ja)
        sc = _composite(ss, sa)
        print(f"  json={jc:.3f}  standalone={sc:.3f}  gap={jc-sc:+.3f}")

    models = [m for m in models if m in j_results]
    models.sort(key=lambda m: _composite(*j_results[m]), reverse=True)

    print(f"\n\n{'='*60}")
    print("GAP TABLE  (JSON - standalone, per metric)")
    print("positive = standalone under-scores  |  negative = over-scores")
    print(f"{'='*60}\n")
    print_table(models, j_results, s_results)

    print(f"\n\n{'='*60}")
    print("OVERALL COMPOSITE  (50% shapes + 50% arrows)")
    print(f"{'='*60}")
    print(f"  {'Model':<50}  {'JSON':>7}  {'Standalone':>10}  {'Gap':>6}")
    print(f"  {'-'*80}")
    for m in models:
        jc = _composite(*j_results[m])
        sc = _composite(*s_results[m])
        print(f"  {m:<50}  {jc:>7.3f}  {sc:>10.3f}  {jc-sc:>+6.3f}")


# ---------------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------------

def _pick_peculiar_samples(gt_svg_dir, n=5):
    if not gt_svg_dir:
        return []
    scored = []
    for svg_f in sorted(gt_svg_dir.glob("*.svg")):
        try:
            root = ET.parse(str(svg_f)).getroot()
        except Exception:
            continue

        score = 0.0
        seen_ids = set()
        for elem in root.iter():
            eid = elem.get("id", "")
            if eid.startswith("shape_") and eid not in seen_ids:
                seen_ids.add(eid)
                ctags = {strip_ns(c.tag) for c in elem}
                if "polygon" in ctags and len(ctags) > 1:
                    score += 3
                if "ellipse" in ctags:
                    score += 2
                if any("translate" in (c.get("transform", "")) for c in elem):
                    score += 2

        patterns  = sum(1 for e in root.iter() if strip_ns(e.tag) == "pattern")
        gradients = sum(1 for e in root.iter()
                        if strip_ns(e.tag) in ("linearGradient", "radialGradient"))
        score += len(seen_ids) * 0.3 + patterns * 2 + gradients * 1.5
        scored.append((score, svg_f.stem))

    scored.sort(reverse=True)
    return [sid for _, sid in scored[:n]]


def _pick_best_model(gt_svg_dir, input_dir):
    if not input_dir.exists():
        return None
    available = {d.name for d in input_dir.iterdir() if d.is_dir()}
    if not available:
        return None
    best, best_score = None, -1.0
    for model in available:
        model_dir = input_dir / model
        ss, sa = evaluate_standalone_model(model_dir, gt_svg_dir)
        score = _composite(ss, sa)
        if score > best_score:
            best, best_score = model, score
    return best


def _html_index(highlights_dir, entries, model_name):
    rows = []
    for e in entries:
        sid = e["sid"]
        rows.append(f"""
  <tr>
    <td><b>{sid}</b></td>
    <td>{e['n_shapes']} shapes / {e['n_arrows']} arrows</td>
    <td>{e['features']}</td>
    <td>{e['shape_score']:.3f}</td>
    <td>{e['arrow_score']:.3f}</td>
    <td>{e['composite']:.3f}</td>
    <td>
      <a href="{sid}_gt.svg">GT</a> &nbsp;
      <a href="{sid}_inference.svg">inference</a> &nbsp;
      <a href="{sid}_shapes_ann.svg">shapes ann</a> &nbsp;
      <a href="{sid}_arrows_ann.svg">arrows ann</a>
    </td>
  </tr>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Highlights</title>
<style>
  body {{ font-family: monospace; padding: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  th {{ background: #eee; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  a {{ color: #0055cc; }}
</style>
</head>
<body>
<h2>Highlights &mdash; 5 peculiar samples ({model_name} scores)</h2>
<p>Annotations rendered on the inference SVG: <b>green dashed</b> = expected shapes,
<b>red dashed</b> = model-detected shapes with per-metric scores.</p>
<table>
  <tr>
    <th>Sample</th><th>Size</th><th>Features</th>
    <th>Shapes</th><th>Arrows</th><th>Composite</th><th>Files</th>
  </tr>
  {''.join(rows)}
</table>
</body></html>"""
    (highlights_dir / "index.html").write_text(html, encoding="utf-8")


def generate_highlights(eval_set=None, n=5):
    gt_svg_dir = _find_gt_svg_dir(eval_set)
    input_dir  = _find_input_dir(eval_set)

    if not gt_svg_dir:
        print("ERROR: GT SVG folder not found."); return

    highlights_dir = BASE / "output" / "highlights"
    highlights_dir.mkdir(parents=True, exist_ok=True)

    sids  = _pick_peculiar_samples(gt_svg_dir, n)
    model = _pick_best_model(gt_svg_dir, input_dir)
    if not model:
        print(f"No models found in {input_dir}"); return

    print(f"Generating highlights for {n} samples using model '{model}'")
    print(f"  Samples: {', '.join(sids)}\n")

    entries = []
    for sid in sids:
        gen_svg = input_dir / model / f"{sid}.svg"
        gt_svg  = gt_svg_dir / f"{sid}.svg"

        if not gen_svg.exists():
            print(f"  {sid}: model SVG not found, skipping"); continue

        shutil.copy2(gt_svg,  highlights_dir / f"{sid}_gt.svg")
        shutil.copy2(gen_svg, highlights_dir / f"{sid}_inference.svg")

        s_sc, s_rep, a_sc, a_rep = evaluate_standalone(
            gen_svg, gt_svg, ann_dir=highlights_dir, label=sid)

        for suffix, dest in [("_shapes.svg", f"{sid}_shapes_ann.svg"),
                              ("_arrows.svg", f"{sid}_arrows_ann.svg")]:
            src = highlights_dir / f"{sid}{suffix}"
            if src.exists():
                src.rename(highlights_dir / dest)

        (highlights_dir / f"{sid}_shapes.txt").write_text(s_rep, encoding="utf-8")
        (highlights_dir / f"{sid}_arrows.txt").write_text(a_rep, encoding="utf-8")

        sc   = s_sc.get("composite", 0)
        ac   = a_sc.get("composite", 0)
        comp = (sc + ac) / 2

        meta     = reconstruct_meta(gt_svg)
        entities = meta.get("entities", {})
        n_shapes = sum(1 for v in entities.values() if v.get("type") != "arrow")
        n_arrows = sum(1 for v in entities.values() if v.get("type") == "arrow")
        feats = set()
        for v in entities.values():
            if v.get("type") == "arrow": continue
            t = v.get("type", "")
            if t.startswith("3d-"):            feats.add(t)
            if v.get("stacked", 1) > 1:        feats.add("stacked")
            fs = v.get("fillStyle", "solid")
            if fs not in ("solid", "none", ""): feats.add(fs)

        entries.append({
            "sid": sid, "n_shapes": n_shapes, "n_arrows": n_arrows,
            "features": ", ".join(sorted(feats)) or "standard",
            "shape_score": sc, "arrow_score": ac, "composite": comp,
        })
        print(f"  {sid}: shapes={sc:.3f}  arrows={ac:.3f}  composite={comp:.3f}")

    _html_index(highlights_dir, entries, model)
    print(f"\nHighlights written to: {highlights_dir}/")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--highlights" in sys.argv:
        # Pull --eval-set from argv manually so highlights works without argparse
        es = None
        for i, a in enumerate(sys.argv):
            if a == "--eval-set" and i + 1 < len(sys.argv):
                es = sys.argv[i + 1]
        generate_highlights(eval_set=es)
    else:
        main()
