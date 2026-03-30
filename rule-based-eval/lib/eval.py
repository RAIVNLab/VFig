#!/usr/bin/env python3
"""Single-pair SVG evaluator — evaluate one generated SVG against one GT SVG.

Useful for debugging a specific sample without running the full model folder.

Usage:
    python lib/eval.py generated.svg gt.svg
    python lib/eval.py generated.svg gt.svg -t shapes_ann.svg arrows_ann.svg
    python lib/eval.py generated.svg gt.svg -v
"""

import sys, argparse
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
ROOT    = LIB_DIR.parent
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(ROOT))   # for main.py (evaluate_standalone)

from main import evaluate_standalone


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single generated SVG against a GT SVG.")
    parser.add_argument("gen_svg",  help="Generated (model output) SVG file")
    parser.add_argument("gt_svg",   help="Ground-truth SVG file")
    parser.add_argument("-t", nargs=2, metavar=("SHAPES_ANN", "ARROWS_ANN"),
                        default=None,
                        help="Write annotated SVGs to these paths")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print the full per-shape/arrow report")
    args = parser.parse_args()

    gen_svg = Path(args.gen_svg)
    gt_svg  = Path(args.gt_svg)

    if not gen_svg.exists():
        print(f"ERROR: {gen_svg} not found"); sys.exit(1)
    if not gt_svg.exists():
        print(f"ERROR: {gt_svg} not found"); sys.exit(1)

    ann_dir = None
    if args.t:
        # evaluate_standalone writes to ann_dir/<sid>_shapes.svg etc.
        # For a single file we use a temp dir and rename.
        import tempfile, shutil
        ann_dir = Path(tempfile.mkdtemp())

    s_sc, s_rep, a_sc, a_rep = evaluate_standalone(
        gen_svg, gt_svg, ann_dir=ann_dir, label=gen_svg.stem)

    if args.verbose:
        print(s_rep)
        print()
        print(a_rep)
        print()

    sc = s_sc.get("composite", 0.0)
    ac = a_sc.get("composite", 0.0)
    print(f"{'='*50}")
    print(f"Sample   : {gen_svg.name}")
    print(f"Shapes   : {sc:.3f}")
    print(f"Arrows   : {ac:.3f}")
    print(f"Overall  : {(sc + ac) / 2:.3f}")
    print(f"{'='*50}")

    if args.t and ann_dir:
        sid = gen_svg.stem
        shapes_out, arrows_out = Path(args.t[0]), Path(args.t[1])
        src_s = ann_dir / f"{sid}_shapes.svg"
        src_a = ann_dir / f"{sid}_arrows.svg"
        if src_s.exists(): shutil.copy2(src_s, shapes_out)
        if src_a.exists(): shutil.copy2(src_a, arrows_out)
        shutil.rmtree(ann_dir, ignore_errors=True)
        print(f"Annotations: {shapes_out}  {arrows_out}")


if __name__ == "__main__":
    main()
