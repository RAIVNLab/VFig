# Rule-Based Diagram Evaluation

Rule-based evaluation framework for scoring model-generated SVG diagrams against ground-truth references. Scores shapes and arrows independently across structural, visual, and connectivity metrics.

---

## Repository Layout

`input/`, `eval-set/`, and `output/` are not committed and must be created locally. Run `python run_all_evals.py --setup` to create the skeleton, then populate with your data.

```
eval-set/                            ← GT SVGs (not committed)
  <eval-set-name>/                   ← GT SVGs placed directly here
input/                               ← model inference SVGs (not committed)
  <eval-set-name>/
    <model_name>/                    ← one folder per model; filenames must match GT stems
output/                              ← written by evaluators (not committed)
  all_eval_results.json              ← consolidated scores across all eval-sets
  batch_results.json                 ← single-eval-set batch output
  <model_name>/                      ← per-model per-sample .txt reports
lib/
  eval_shapes.py                     ← shape evaluation library
  eval_arrows.py                     ← arrow evaluation library
  eval.py                            ← single SVG-pair evaluator (debug)
  compare_evals.py                   ← JSON-backed vs. standalone comparison + highlights
  utils.py                           ← shared color/label/geometry helpers
main.py                              ← evaluate one model against a GT SVG folder
batch_main.py                        ← batch evaluator for one eval-set
run_all_evals.py                     ← evaluate all eval-sets and models; produce summary
check_svgs.py                        ← validate SVG files for XML well-formedness
```

---

## Quick Start

```bash
# 1. Create the expected directory layout
python run_all_evals.py --setup

# 2. Populate eval-set/<name>/ with GT SVGs
#    and input/<eval-set>/<model>/ with model inference SVGs

# 3. Evaluate all eval-sets and all models; write consolidated JSON + print tables
python run_all_evals.py --superset

# Evaluate one model on one eval-set
python main.py <model> --gt-svg eval-set/<name> --input-dir input/<eval-set>

# Evaluate all models in one eval-set (batch)
python batch_main.py --gt-svg eval-set/<name>

# Print tables from a previously saved JSON without re-evaluating
python run_all_evals.py --tables-only

# Load scores from the JSON and read per-metric breakdowns
python -c "import json; d=json.load(open('output/all_eval_results.json')); ..."
```

---

## main.py — single model

```
python main.py <model_folder>                             # reads from input/<model>
python main.py <model_folder> -v                          # verbose per-sample output
python main.py <model_folder> -t                          # write annotation SVGs to output/
python main.py -d                                         # dry run: GT SVG as input (~1.0)
python main.py <model_folder> --gt-svg PATH               # explicit GT SVG folder
python main.py <model_folder> --input-dir PATH            # model SVGs in PATH/<model>
python main.py <model_folder> --input-dir PATH --gt-svg PATH
```

Reconstructs ground-truth metadata directly from the GT SVG (no separate JSON needed).
Writes per-sample `.txt` score reports to `output/<model>/`.

---

## batch_main.py — one eval-set, all models

```
python batch_main.py                              # all models in input/
python batch_main.py --models <model1> <model2>    # restrict to specific models
python batch_main.py --gt-svg path/to/svgs        # explicit GT SVG folder
python batch_main.py -v                           # verbose per-sample errors
```

Evaluates every model in `input/` across all GT samples. Models missing a sample receive 0.0. Saves `output/batch_results.json` with full per-metric per-model scores.

---

## run_all_evals.py — all eval-sets

```
python run_all_evals.py --setup                      # create input/, eval-set/, output/ layout
python run_all_evals.py                              # all eval-sets, all models
python run_all_evals.py <eval-set1> <eval-set2>       # specific eval-sets only
python run_all_evals.py --models <model1> <model2>    # restrict models
python run_all_evals.py --out results.json            # custom output path
python run_all_evals.py --tables-only                 # print summary tables from saved JSON
python run_all_evals.py --tables-only --out x.json    # load from specific JSON
python run_all_evals.py --superset                    # also compute superset scores
```

Discovers eval-sets from `input/` subfolders, matches each to the GT SVG folder in `eval-set/`, evaluates every model, and saves `output/all_eval_results.json`.

**Superset scoring:** The `--superset` flag computes scores over only the "clean" samples that every non-high-failure model successfully evaluated. Models with >51% missing or errored samples are excluded from the superset intersection and receive 0s. The superset N is logged in the summary table.

Eval-sets and models excluded by default from summary tables are configured via `EXCLUDED_EVAL_SETS` and `EXCLUDED_MODELS` constants at the top of `run_all_evals.py`.

---

## lib/compare_evals.py — JSON-backed comparison

```
python lib/compare_evals.py                                    # default eval-set, all models
python lib/compare_evals.py <model1> <model2>                   # specific models
python lib/compare_evals.py --eval-set <name>                  # specific eval-set
python lib/compare_evals.py --eval-set <name> <model>          # eval-set + specific models
python lib/compare_evals.py --highlights                        # generate 5 peculiar sample highlights
```

Compares JSON-backed evaluation (using `eval-set/objects/*.json` ground-truth metadata) against standalone SVG reconstruction (`main.py` method). The delta shows where SVG reconstruction diverges from ground truth. `--highlights` selects 5 structurally complex GT samples, evaluates the best model, and writes annotated SVGs + `index.html` to `output/highlights/`.

---

## lib/eval.py — single SVG pair (debug)

```
python lib/eval.py generated.svg gt.svg
python lib/eval.py generated.svg gt.svg -v
python lib/eval.py generated.svg gt.svg -t shapes_ann.svg arrows_ann.svg
```

Useful for debugging a specific sample. Prints shape/arrow/overall composite.

---

## lib/eval_shapes.py / lib/eval_arrows.py — low-level

```
python lib/eval_shapes.py generated.svg gt.svg metadata.json
python lib/eval_shapes.py generated.svg gt.svg metadata.json --annotate out.svg

python lib/eval_arrows.py generated.svg gt.svg metadata.json
```

Require the metadata JSON. Use `lib/eval.py` or `main.py` when you only have SVGs.

---

## check_svgs.py — validate SVG files

```
python check_svgs.py <dir>            # check all .svg files under <dir>
python check_svgs.py <dir> --verbose  # also print OK files
```

Validates SVG files using stdlib only (`xml.etree.ElementTree`). Checks XML well-formedness and that the root element is `<svg>`. Exits with code 1 if any files are invalid.

---

## Reading Results from JSON

`output/all_eval_results.json` is the canonical results file. Structure:

```python
{
  "<eval-set-name>": {
    "<model-name>": {
      "overall":   float,          # 0.5 * shapes.composite + 0.5 * arrows.composite
      "shapes": {
        "label":        float,
        "type":         float,
        "fill_color":   float,
        "fill_style":   float,
        "stroke_color": float,
        "border_style": float,
        "position":     float,
        "font":         float,
        "aspect_ratio": float,
        "extra_penalty":float,
        "composite":    float
      },
      "arrows": {
        "source":       float,
        "dest":         float,
        "head":         float,
        "head_size":    float,
        "curve":        float,
        "color":        float,
        "overlap":      float,
        "extra_penalty":float,
        "composite":    float
      },
      "missing_count":         int,
      "errored_count":         int,
      "missing_samples":       [str, ...],
      "superset_overall":      float,   # score over clean shared sample subset
      "superset_shapes":       {...},   # same keys as shapes above
      "superset_arrows":       {...},   # same keys as arrows above
      "superset_n":            int,     # number of samples in the superset
      "superset_high_failure": bool     # true if model excluded from superset
    }
  }
}
```

### Example: print overall scores for all models on one eval-set

```python
import json

results = json.load(open("output/all_eval_results.json"))
eval_set = list(results.keys())[0]   # or name the eval-set explicitly

for model, data in results[eval_set].items():
    print(f"{model:20s}  overall={data['overall']:.3f}")
```

### Example: per-metric shape breakdown for a specific eval-set

```python
import json

results = json.load(open("output/all_eval_results.json"))

eval_set = list(results.keys())[0]   # or name the eval-set explicitly
shape_metrics = ["label", "type", "fill_color", "fill_style", "stroke_color",
                 "border_style", "position", "font", "aspect_ratio", "composite"]

# Header
print(f"{'Metric':<16}" + "".join(f"  {m[:12]:>12}" for m in results[eval_set]))
# Rows
for metric in shape_metrics:
    row = f"{metric:<16}"
    for model, data in results[eval_set].items():
        row += f"  {data['shapes'][metric]:>12.3f}"
    print(row)
```

### Example: compare superset scores across eval-sets

```python
import json

results = json.load(open("output/all_eval_results.json"))

eval_sets = list(results.keys())
all_models = sorted({m for es in results.values() for m in es})

print(f"{'Model':<20}" + "".join(f"  {es[:14]:>14}" for es in eval_sets))
for model in all_models:
    row = f"{model:<20}"
    for es in eval_sets:
        entry = results.get(es, {}).get(model, {})
        v = entry.get("superset_overall")
        hf = entry.get("superset_high_failure", False)
        if v is None:
            row += f"  {'—':>14}"
        elif hf:
            row += f"  {'[excl]':>14}"
        else:
            row += f"  {v:>14.3f}"
    print(row)
```

### Example: find best model per eval-set

```python
import json

results = json.load(open("output/all_eval_results.json"))

for eval_set, models in results.items():
    best = max(models.items(), key=lambda kv: kv[1]["overall"])
    print(f"{eval_set}: best={best[0]}  overall={best[1]['overall']:.3f}")
```

---

## Overall Score

```
Overall = 0.5 × shape_composite + 0.5 × arrow_composite
```

Both composites are in [0, 1]. See `metrics.md` for full metric definitions.

---

## Shape Metrics

Ground-truth shapes are identified in the model SVG by matching `<text>` labels. Nine metrics scored with equal weight (≈11% each):

| Metric         | Description                                           |
| -------------- | ----------------------------------------------------- |
| `label`        | Normalized text label similarity                      |
| `type`         | SVG element tag vs. GT shape type                     |
| `fill_color`   | RGB Euclidean distance (normalized to [0,1])          |
| `fill_style`   | solid / gradient / pattern match                      |
| `stroke_color` | RGB Euclidean distance                                |
| `border_style` | solid / dashed / dotted / dash-dot                    |
| `position`     | Relative offset from anchor shape (direction + ratio) |
| `font`         | Font family exact or same-class match                 |
| `aspect_ratio` | Width/height ratio similarity                         |

Extra (hallucinated) shapes apply a precision penalty:

```
shape_composite = raw_mean × (1 - n_extra / (n_gt + n_extra))
```

Shape detection includes `rect`, `circle`, `ellipse`, `polygon` elements with solid fills, **and** those with `fill="none"` but a visible stroke (container/border boxes). `polyline` and `path` elements with `fill="none"` are excluded (typically connectors or arrows).

---

## Arrow Metrics

Arrows are matched from open `<path>` or `<line>` elements with a stroke and no fill. Seven metrics, equal weight (≈14%):

| Metric      | Description                                             |
| ----------- | ------------------------------------------------------- |
| `source`    | Start endpoint nearest to correct source shape          |
| `dest`      | End endpoint nearest to correct destination shape       |
| `head`      | Arrowhead present (`marker-end` or `<polygon>` sibling) |
| `head_size` | Rendered arrowhead width vs. GT width ratio             |
| `curve`     | Straight / curved match                                 |
| `color`     | RGB stroke color distance                               |
| `overlap`   | Endpoints not penetrating wrong shapes                  |

Missing GT arrows score 0.0. Extra arrows apply the same precision penalty as shapes.

---

## GT SVG Reconstruction

`main.py` reconstructs ground-truth metadata from the GT SVG without any JSON file. Reconstruction accuracy over 500 samples:

| Component              | Accuracy  |
| ---------------------- | --------- |
| Shape type             | 0.947     |
| Shape bounds           | 0.937     |
| Label                  | 0.906     |
| Font                   | 0.907     |
| Fill style             | 0.928     |
| Fill color             | 0.925     |
| Stroke color           | 0.928     |
| Border style           | 0.992     |
| **Shape overall**      | **0.931** |
| Arrow from/to          | 0.999     |
| Arrow color/style/dash | 1.000     |
| **Arrow overall**      | **0.999** |
| **Overall**            | **0.955** |

Known limitations: `3d-prism` and `3d-cube` produce identical SVG structure and cannot be distinguished. Circles rendered as cubic-bezier paths with a non-square bounding box are reconstructed as `ellipse` (the evaluator accepts both interchangeably).

---

## Annotation SVGs

Running with `-t` writes `{sid}_shapes.svg` and `{sid}_arrows.svg` overlaid on the model output:

- **Green dashed boxes** — expected GT shapes with labels
- **Red dashed boxes** — model-detected shapes with per-metric scores (green ≥ 0.8, orange ≥ 0.5, red < 0.5)

---

## Output Files

| File                              | Contents                                                                |
| --------------------------------- | ----------------------------------------------------------------------- |
| `output/all_eval_results.json`    | Consolidated per-metric scores for all eval-sets and models             |
| `output/<model>/{sid}_shapes.txt` | Per-shape score breakdown                                               |
| `output/<model>/{sid}_arrows.txt` | Per-arrow score breakdown                                               |
| `output/<model>/{sid}_shapes.svg` | Annotated shape overlay (with `-t`)                                     |
| `output/<model>/{sid}_arrows.svg` | Annotated arrow overlay (with `-t`)                                     |
| `output/batch_results.json`       | Legacy single-eval-set batch output                                     |
| `output/highlights/`             | 5 peculiar samples with GT, inference, annotated SVGs, and `index.html` |
