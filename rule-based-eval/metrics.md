# Evaluation Metrics

Each metric produces a score in **[0.0 – 1.0]**. Shape and arrow composites are averaged equally to form the **overall score**: `overall = (shape_composite + arrow_composite) / 2`.

---

## Color scoring — justified normalisation constant

All colour comparisons use the Euclidean L2 distance in integer RGB space, normalised by the **theoretical maximum**:

$$\text{color-score}(c_1, c_2) = \max\!\left(0,\ 1 - \frac{\|c_1 - c_2\|_2}{255\sqrt{3}}\right)$$

The divisor $255\sqrt{3} \approx 441.67$ is the exact maximum distance between any two points in the cube $[0,255]^3$ (black vs. white), so dividing by it maps any colour pair to $[0, 1]$ with a principled upper bound. This is not an empirical magic number — it is the $\ell_2$ diameter of the RGB space.

---

## Shape Metrics

### `label`
**Extraction:** The text content of each extracted SVG shape (group label or nearby text element) is normalised to lowercase with collapsed whitespace and compared to the ground-truth label from the JSON metadata.
**Grade:** 1.0 for exact match; proportional word-overlap ratio `|common_words| / max(|gen_words|, |gt_words|)` for partial matches; 0.0 for no words in common.

### `type`
**Extraction:** The SVG element tag of the best-matching shape (`rect`, `ellipse`, `circle`, `polygon`, `path`, `g`) is looked up in a hand-coded allowable-tags table keyed by the ground-truth shape type (e.g. `rectangle → {rect}`, `ellipse → {ellipse, circle}`, `3d-cube → {g, rect, polygon}`).
**Grade:** 1.0 for an accepted tag; 0.3–0.4 for a plausible substitute (e.g. `path` for `rectangle`); 0.0 for a completely wrong element.

### `fill_color`
**Extraction:** The ground-truth colour is read directly from the `fillColor` field of the JSON metadata (avoiding SVG-extraction errors for 3D and pattern-filled shapes). The model colour is the `fill` attribute resolved from the generated SVG element or its gradient stop. Both are parsed to RGB tuples via `_parse_color`, which handles `#rrggbb`, `#rgb`, CSS named colours, and `rgb()` / `rgba()`.
**Grade:** $1 - \|c_\text{gt} - c_\text{gen}\|_2 / (255\sqrt{3})$; 1.0 = identical colour, 0.0 = maximally distant (black vs. white).

### `fill_style`
**Extraction:** The ground-truth `fillStyle` is read from JSON. The model fill style is inferred by inspecting the SVG `<defs>` section: patterns with `<circle>` children → `dots`; two or more crossing `<line>` directions → `crosshatch`; single-direction lines → `hatching`; `<linearGradient>` / `<radialGradient>` → corresponding gradient; direct solid fill → `solid`.
**Grade:** 1.0 for exact category match; 0.5 if both are non-solid but different pattern types (e.g. `dots` vs. `hatching`); 0.0 if one side is solid and the other is patterned.

### `stroke_color`
**Extraction:** Same RGB-distance pipeline as `fill_color`, applied to the ground-truth `strokeColor` JSON field vs. the `stroke` attribute on the generated shape element.
**Grade:** $1 - \|c_\text{gt} - c_\text{gen}\|_2 / (255\sqrt{3})$; same scale as fill colour.

### `border_style`
**Extraction:** The ground-truth `borderStyle` is read from JSON. The model border style is inferred from the `stroke-dasharray` attribute: absent or `none` → `solid`; gap value ≤ 3 → `dotted`; single longer gap → `dashed`; two-pair sequence (e.g. `8,3,2,3`) → `dash-dot`.
**Grade:** 1.0 for exact match; 0.5 if both are non-solid dashes of different types; 0.0 if one is solid and the other is not.

### `position`
**Extraction:** One matched shape is selected as the anchor (whichever maximises the mean relative-position score across all others). For each other shape, the direction angle and unit-vector from the anchor to the shape are computed in both GT and generated layouts, then compared as a direction score (angle difference penalised linearly over 0–180°) and an offset-vector cosine-like score.
**Grade:** $0.5 \times \text{direction} + 0.5 \times \text{offset-vector}$, both in $[0,1]$; anchor always scores 1.0; unmatched shapes score 0.0.

### `font`
**Extraction:** The first `font-family` token is extracted from the ground-truth JSON `font` field and from the generated SVG's `font-family` attribute (searched on the element, `<tspan>` children, then parent groups). Both are classified into serif / sans-serif / monospace using a fixed vocabulary (e.g. `{times, georgia, palatino, garamond}` → serif).
**Grade:** 1.0 for an exact first-family name match; 0.5 if both map to the same class; 0.0 for a class mismatch or missing font attribute.

### `aspect_ratio`
**Extraction:** Width-to-height ratio is computed from the bounding boxes of the ground-truth entity (from JSON `bounds`) and the generated shape. The ratio $r = \max(AR_\text{gt}/AR_\text{gen},\, AR_\text{gen}/AR_\text{gt}) \geq 1$ measures relative deviation.
**Grade:** 1.0 if $r \leq 1.2$; linearly decays via $1 - (r-1.2)/1.8$ to 0.0 at $r = 3.0$. The 1.2 lower bound allows for minor rendering differences; 3.0 corresponds to a shape that is three times as wide (or tall) as the GT.

### `extra_penalty`
**Extraction:** After matching, any generated shapes whose labels do not match any ground-truth label are counted as `n_extra`.
**Grade:** Penalty $= n_\text{extra} / (n_\text{gt} + n_\text{extra})$; composite is multiplied by $(1 - \text{penalty})$. This is the fraction of all detected shapes that are hallucinated — a natural precision-based penalty.

### `composite` (shapes)
**Extraction:** Unweighted mean of the nine per-shape metric scores across all matched GT shapes.
**Grade:** Raw mean scaled by $(1 - \text{extra-penalty})$; each metric contributes $\approx 11\%$ of the composite.

---

## Arrow Metrics

### `source`
**Extraction:** The generated arrow's start endpoint is matched to the nearest detected shape bounding box (within 100 px). The detected shape's normalised label is compared to the ground-truth `from` label. The matching also tries the arrow reversed and keeps the higher-scoring orientation.
**Grade:** 1.0 if the closest detected shape matches the expected source; 0.0 otherwise.

### `dest`
**Extraction:** Same proximity-matching pipeline applied to the arrow's end endpoint vs. the ground-truth `to` label.
**Grade:** 1.0 for correct destination; 0.0 otherwise.

### `head`
**Extraction:** An arrowhead is considered present if the path or line element has a `marker-end` attribute referencing a `<marker>` in `<defs>`, or if a `<polygon>` sibling exists in the same group.
**Grade:** Binary — 1.0 if present, 0.0 if absent.

### `head_size`
**Extraction:** Both sides are converted to **stroke-width-relative units** before comparison (GT polygon width ÷ GT stroke-width; model marker viewport height, already in sw-units when `markerUnits=strokeWidth`). The ratio $r = \max(v_\text{model}/v_\text{gt},\, v_\text{gt}/v_\text{model}) \geq 1$ is then used.
**Grade:** Three tolerance tiers: $r \leq 1.3$ → 1.0 (±30%: within normal SVG renderer rounding); $r \leq 1.8$ → 0.6 (±80%: noticeably off but same visual class); $r \leq 2.5$ → 0.3 (±150%: clearly disproportionate); $r > 2.5$ → 0.1.

### `curve`
**Extraction:** The ground-truth `style` field (`straight` or `curved`) is compared to whether the generated path contains any Bezier or arc commands (`Q`, `C`, `S`, `T`, `A`).
**Grade:** Binary — 1.0 for matching curvature, 0.0 for mismatch; no partial credit since straight vs. curved is a discrete structural property.

### `color`
**Extraction:** The ground-truth arrow `color` field is compared to the `stroke` on the generated path or line element, with full CSS inheritance resolved from parent `<g>` elements and class-based style blocks.
**Grade:** $1 - \|c_\text{gt} - c_\text{gen}\|_2 / (255\sqrt{3})$; same formula as shape colour.

### `overlap`
**Extraction:** Each arrow endpoint is checked against every detected shape bounding box. Endpoints that land strictly inside a shape they are *not* supposed to connect to are counted as violations.
**Grade:** 1.0 for zero violations; 0.4 for one bad endpoint; 0.0 for two or more.

### `extra_penalty`
**Extraction:** Generated arrows not matched to any ground-truth arrow are counted as `n_extra`.
**Grade:** Same precision-based formula as shapes: $(1 - n_\text{extra}/(n_\text{gt} + n_\text{extra}))$ applied as a multiplier on the raw arrow composite.

### `composite` (arrows)
**Extraction:** Unweighted mean of the seven per-arrow metric scores across all matched GT arrows.
**Grade:** Raw mean scaled by $(1 - \text{extra-penalty})$; each metric contributes $\approx 14\%$ of the composite.
