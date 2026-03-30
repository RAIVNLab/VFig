#!/usr/bin/env python3
"""Evaluate a model output folder against ground-truth SVGs.

Parses each GT SVG to reconstruct the ground-truth metadata on the fly —
no separate JSON metadata file required.

Usage:
    python main.py <model_folder>                      # reads SVGs from input/<model>
    python main.py <model_folder> -t                   # also write annotation SVGs to output/
    python main.py <model_folder> -v                   # verbose per-sample output
    python main.py -d                                  # dry run: GT SVG as input (~1.0)
    python main.py <model_folder> --gt-svg PATH                     # explicit GT SVG folder
    python main.py <model_folder> --input-dir PATH                  # model SVGs in PATH/<model>
    python main.py <model_folder> --input-dir PATH --gt-svg PATH    # fully explicit

LIMITATIONS
-----------
1.  Shape detection relies on id="shape_N" attributes in the GT SVG.  If the
    GT SVG has no such IDs, a fallback heuristic scans for shape-like elements
    in document order and assigns IDs automatically.  The fallback is less
    reliable: it may pick up background rects, decorative elements, or skip
    complex nested groups.

2.  Arrow detection relies on either:
      a) a <!-- Arrows --> … <!-- Shapes --> comment block (eval-set format), or
      b) root-level <g> elements that contain a shaft (line/path) AND a polygon
         arrowhead as direct siblings.
    Model SVGs that use marker-end="url(#arrowhead)" on standalone lines/paths
    do NOT match either pattern — the evaluation will find 0 arrows.  This only
    affects evaluation of arrow metrics; shape metrics are unaffected.

3.  Label assignment gives strict priority to <text> elements whose position
    falls strictly inside the shape bounding box, then falls back to proximity
    (nearest unused <text> within 150 px).  Labels far from their shape, or
    layouts where multiple shapes overlap heavily, may still be incorrectly
    matched.

4.  3d-prism and 3d-cube produce identical SVG structure (two rects + two
    polygons) and cannot be distinguished from the SVG alone.  Both will be
    reconstructed as 3d-cube.

5.  Circles rendered as cubic-bezier <path> elements with a slightly non-square
    bounding box (aspect ratio > 1.15) will be reconstructed as ellipse.  The
    evaluator accepts both types as interchangeable, so this does not affect
    evaluation scores.

6.  CSS class rules in <style> blocks are fully supported by the model-SVG
    evaluator (lib/eval_shapes.py).  The GT SVG reconstructor in this file
    reads inline attributes only, but GT SVGs in our eval-set always use
    inline attributes so this has no practical effect on accuracy.

7.  Shapes in model output SVGs placed inside nested
    <g transform="translate(...)"> containers have positions relative to the
    group transform.  lib/eval_shapes.py does not accumulate parent transforms
    when computing bounding boxes, so shape positions may be misread for
    deeply nested layouts.
"""

import sys, re, json, math, tempfile, argparse, shutil
from pathlib import Path
from collections import defaultdict
from xml.etree import ElementTree as ET

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "lib"))

from utils import (strip_ns, safe_float, find_nums, parse_translate,
                   normalize_label, color_score, canvas_size,
                   ZERO_SHAPE_SCORES, ZERO_ARROW_SCORES)
from eval_shapes import (
    bbox, parse_defs, detect_fill_style, detect_border_style,
    parse_gradient_colors, resolve_fill_color, collect_text, get_font,
    SHAPE_TAGS, evaluate_shapes_file,
)
from eval_arrows import (
    evaluate_arrows_file, path_start, path_end, detect_curvature,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_DIR  = BASE / "input"
OUTPUT_DIR = BASE / "output"


def _find_gt_svg_dir():
    """Auto-detect the GT SVG folder."""
    for c in [BASE / "eval-set"]:
        if c.exists() and list(c.glob("*.svg")):
            return c
    return None


# ---------------------------------------------------------------------------
# Fill color helpers  (resolving url(#id) references into plain hex strings)
# ---------------------------------------------------------------------------

def _build_fill_color_map(root):
    """scan <defs> and return {id: hex_color} for every gradient and pattern.

    gradients  → first stop-color (the dominant/base fill color)
    patterns   → fill of the first <rect> child (the background tile color)
    """
    colors = {}
    for elem in root.iter():
        tag = strip_ns(elem.tag)

        if tag in ("linearGradient", "radialGradient"):
            eid = elem.get("id", "")
            if not eid:
                continue
            for stop in elem:
                if strip_ns(stop.tag) != "stop":
                    continue
                # stop-color may live in the style attribute instead of its own attr
                sc = stop.get("stop-color", "")
                if not sc:
                    style = stop.get("style", "")
                    m = re.search(r"stop-color\s*:\s*([^;]+)", style)
                    sc = m.group(1).strip() if m else ""
                if sc and sc.lower() not in ("none", ""):
                    colors[eid] = sc
                    break  # only need the first non-transparent stop

        elif tag == "pattern":
            eid = elem.get("id", "")
            if not eid:
                continue
            for child in elem:
                if strip_ns(child.tag) == "rect":
                    fc = child.get("fill", "")
                    if fc and fc.lower() not in ("none", ""):
                        colors[eid] = fc
                    break  # first rect is the background color tile

    return colors


def _resolve_fill_color(fill_attr, fill_color_map):
    """turn a fill attribute (possibly url(#id)) into a plain hex string."""
    if not fill_attr:
        return ""
    if fill_attr.startswith("url(#"):
        ref = fill_attr[5:-1]  # strip url(# and )
        return fill_color_map.get(ref, fill_attr)
    return fill_attr


# ---------------------------------------------------------------------------
# Shape type inference
# (helpers first, public entry point `infer_shape_type` at the bottom)
# ---------------------------------------------------------------------------

def _count_stacked_layers(elem):
    """count depth shadow layers — direct <g translate> children of a shape group."""
    return sum(
        1 for c in elem
        if strip_ns(c.tag) == "g" and "translate" in c.get("transform", "").lower()
    )


def _main_child(elem):
    """for a stacked <g>, find the primary (non-offset) shape element.

    stacked shapes look like:
        <g id="shape_N">
          <g transform="translate(4,4)">  <- shadow copy, offset
            <rect .../>
          </g>
          <rect .../>                     <- main element, no transform
        </g>
    we want the element without a transform attribute.
    """
    # prefer a direct shape child with no transform (the visible front face)
    for c in elem:
        ctag = strip_ns(c.tag)
        if ctag in SHAPE_TAGS and not c.get("transform", ""):
            return c, ctag
    # fallback: look one level deeper inside a translate group
    for c in elem:
        if strip_ns(c.tag) == "g":
            for cc in c:
                cctag = strip_ns(cc.tag)
                if cctag in SHAPE_TAGS:
                    return cc, cctag
    return elem, "g"  # give up — treat whole group as unknown


def _poly_has_horizontal_edge(pts_str, tol=5.0):
    """return true if any two adjacent polygon vertices share the same y (within tol).

    used to distinguish 3d-trapezoid (has a flat top/bottom) from
    3d-diamond (all vertices at different heights).
    """
    nums = find_nums(pts_str)
    pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    n = len(pts)
    for i in range(n):
        if abs(pts[i][1] - pts[(i + 1) % n][1]) < tol:
            return True
    return False


def _infer_type_direct(elem, tag):
    """infer the json type string for a primitive (non-group) svg element."""
    if tag == "rect":
        w = safe_float(elem.get("width", 0))
        h = safe_float(elem.get("height", 0))
        # square = aspect ratio within 10% of 1:1
        return "square" if h > 0 and abs(w / h - 1.0) <= 0.10 else "rectangle"

    if tag == "circle":
        return "circle"

    if tag == "ellipse":
        rx = safe_float(elem.get("rx", 0))
        ry = safe_float(elem.get("ry", 0))
        # treat equal-radius ellipses as circles
        return "circle" if rx > 0 and ry > 0 and abs(rx / ry - 1.0) <= 0.10 else "ellipse"

    if tag == "polygon":
        return "polygon"

    if tag == "path":
        d = elem.get("d", "")
        n_rel_arc = len(re.findall(r"a", d))   # lowercase 'a' = relative arc = cloud bump
        n_cubic   = len(re.findall(r"[Cc]", d))
        n_abs_arc = len(re.findall(r"A", d))   # uppercase 'A' = absolute arc

        # cloud outlines use many small relative arcs (bumpy silhouette)
        if n_rel_arc >= 3:
            return "cloud"

        # circles/ellipses are often approximated by 4+ cubic bezier segments
        if n_cubic >= 4:
            b = bbox(elem, tag)
            if b:
                w, h = b[2] - b[0], b[3] - b[1]
                if h > 0 and abs(w / h - 1.0) <= 0.15:
                    return "circle"
                return "ellipse"

        # explicit arc commands also form circles/ellipses
        if n_abs_arc >= 2:
            b = bbox(elem, tag)
            if b:
                w, h = b[2] - b[0], b[3] - b[1]
                if h > 0 and abs(w / h - 1.0) <= 0.15:
                    return "circle"
                return "ellipse"

        # paths with many curve commands that don't look round → cloud
        if len(re.findall(r"[QqCcAa]", d)) >= 4:
            return "cloud"

        return "polygon"

    return "rectangle"  # unknown element — default to rectangle


def _infer_type_3d(elem):
    """infer the 3d shape type for an un-stacked <g> (i.e., a 3d rendered shape).

    3d shapes are drawn as multiple face polygons/rects inside a single <g>.
    the distinguishing rules are based on child element types and vertex counts.

    note: 3d-prism and 3d-cube both use rect+polygon children and are
    indistinguishable here — both will be returned as 3d-cube (see LIMITATIONS).
    """
    child_tags = {strip_ns(c.tag) for c in elem}

    # cylinder: has an ellipse or circle as one of its face elements
    if "ellipse" in child_tags or "circle" in child_tags:
        return "3d-cylinder"

    # cube/prism: has at least one rect (front face)
    if "rect" in child_tags:
        return "3d-cube"  # prism looks identical at this level — see LIMITATIONS

    polys = [c for c in elem if strip_ns(c.tag) == "polygon"]
    if polys:
        # hexagon front face has 6 vertices; use max across all face polys
        max_verts = max(len(find_nums(p.get("points", ""))) // 2 for p in polys)
        if max_verts >= 5:
            return "3d-hexagon"

        # trapezoid: front face polygon has a flat horizontal top and bottom edge
        # diamond: all four vertices at different heights (rotated square)
        for p in polys:
            if _poly_has_horizontal_edge(p.get("points", "")):
                return "3d-trapezoid"
        return "3d-diamond"

    # path-only group — most likely a cylinder drawn with arcs
    if "path" in child_tags:
        return "3d-cylinder"

    return "3d-cube"  # catch-all


def infer_shape_type(elem, tag):
    """top-level dispatch: infer json 'type' from an svg element and its tag."""
    if tag != "g":
        # primitive element — direct inference
        return _infer_type_direct(elem, tag)

    n_stacked = _count_stacked_layers(elem)
    if n_stacked > 0:
        # stacked group: find the main (non-shadow) child and infer its type
        main, mtag = _main_child(elem)
        return _infer_type_direct(main, mtag)

    # un-stacked group = 3d shape (multiple face elements)
    return _infer_type_3d(elem)


# ---------------------------------------------------------------------------
# Bounding-box extraction
# ---------------------------------------------------------------------------

def _shape_bounds(elem, tag, defs_map=None):
    """return [x1, y1, x2, y2] for the visible (non-shadow) part of a shape.

    for stacked shapes this is the main front-face element's bbox, not the
    union of all layers (which would include the offset shadow).
    defs_map: optional {id: element} for resolving <use> references.
    """
    if tag == "use":
        if defs_map:
            from eval_shapes import _bbox_use
            b = _bbox_use(elem, defs_map)
            return list(b) if b else [0, 0, 0, 0]
        return [0, 0, 0, 0]

    if tag != "g":
        b = bbox(elem, tag)
        return list(b) if b else [0, 0, 0, 0]

    # if it contains <use> children (no direct SHAPE_TAGS), resolve via defs
    if defs_map:
        direct_shapes = [c for c in elem if strip_ns(c.tag) in SHAPE_TAGS]
        if not direct_shapes:
            from eval_shapes import _bbox_group_resolve_uses
            b = _bbox_group_resolve_uses(elem, defs_map)
            if b:
                return list(b)

    # determine whether this <g> is stacked (has translate-offset shadow layers)
    has_stack = any(
        "translate" in c.get("transform", "").lower()
        for c in elem if strip_ns(c.tag) == "g"
    )

    if has_stack:
        # stacked shape: find the main (non-offset) child and use its bounds
        main, mtag = _main_child(elem)
        container = main if mtag == "g" else None
        if mtag != "g":
            b = bbox(main, mtag)
            if b:
                return list(b)
    else:
        # not stacked: either a 3d shape (multiple face children) or simple <g> wrapper
        main, mtag = _main_child(elem)
        if mtag != "g":
            # check if there are multiple shape siblings — if so, union all
            shape_kids = [c for c in elem if strip_ns(c.tag) in SHAPE_TAGS]
            if len(shape_kids) == 1:
                b = bbox(main, mtag)
                if b:
                    return list(b)
        # 3d shape (or stacked main is also a <g>): union all direct shape children
        container = elem

    # union bounds of all shape-element children in the chosen container
    merged = None
    if container is not None:
        for c in container:
            ctag = strip_ns(c.tag)
            if ctag in SHAPE_TAGS:
                b = bbox(c, ctag)
            else:
                continue
            if not b:
                continue
            if merged is None:
                merged = list(b)
            else:
                merged[0] = min(merged[0], b[0])
                merged[1] = min(merged[1], b[1])
                merged[2] = max(merged[2], b[2])
                merged[3] = max(merged[3], b[3])

    return merged or [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Text / label extraction
# ---------------------------------------------------------------------------

def _collect_texts(root):
    """collect all <text> elements from the svg with their positions and fonts.

    Applies cumulative translate() offsets from ancestor <g> elements so that
    texts inside <g transform="translate(...)"> groups get correct positions.
    """
    parent_map = {c: p for p in root.iter() for c in p}
    texts = []
    skip_tags = {"defs", "style", "title", "desc", "marker", "filter",
                 "linearGradient", "radialGradient", "pattern", "symbol"}

    def _walk(node, ox=0.0, oy=0.0):
        tag = strip_ns(node.tag)
        if tag in skip_tags:
            return
        if tag == "text":
            x = safe_float(node.get("x", 0))
            y = safe_float(node.get("y", 0))
            content = collect_text(node).strip()
            if content:
                texts.append({
                    "x":    x + ox,
                    "y":    y + oy,
                    "text": content,
                    "font": get_font(node, parent_map),
                })
            return
        for child in node:
            child_tag = strip_ns(child.tag)
            if child_tag == "g":
                tx, ty = parse_translate(child.get("transform", ""))
                _walk(child, ox + tx, oy + ty)
            else:
                _walk(child, ox, oy)

    _walk(root)
    return texts


def _collect_label_texts(b, texts, used, margin=50):
    """find and merge all text lines that belong to the shape at bbox b.

    returns (merged_label, font, set_of_consumed_indices).

    multi-line labels (e.g. "Batch Norm\\nLogger") are split across multiple
    <text> nodes stacked vertically.  we merge them by collecting nearby text
    nodes at roughly the same x position and within 40 px y-spacing of each other.
    """
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    # gather candidate text nodes: inside bbox (with margin) or within 250px of centre
    nearby = []
    for i, t in enumerate(texts):
        if i in used:
            continue
        inside = (b[0] - margin <= t["x"] <= b[2] + margin and
                  b[1] - margin <= t["y"] <= b[3] + margin)
        d = math.sqrt((t["x"] - cx) ** 2 + (t["y"] - cy) ** 2)
        # texts strictly inside the bbox get sort priority -1 so they always win
        if b[0] <= t["x"] <= b[2] and b[1] <= t["y"] <= b[3]:
            nearby.append((-1.0, i))
        elif inside or d < 150:  # reduced from 250 to avoid stealing from adjacent shapes
            nearby.append((d, i))

    if not nearby:
        return "", "", set()

    nearby.sort()
    best_i = nearby[0][1]
    bx = texts[best_i]["x"]   # anchor x for multi-line alignment
    indices = [best_i]

    # collect additional lines at same x (±30 px) within 40 px vertical spacing
    for _, i in nearby[1:]:
        t = texts[i]
        if abs(t["x"] - bx) < 30:
            if any(abs(t["y"] - texts[mi]["y"]) < 40 for mi in indices):
                indices.append(i)

    indices.sort(key=lambda i: texts[i]["y"])  # top to bottom
    label = " ".join(texts[i]["text"] for i in indices)
    font  = texts[indices[0]]["font"] or ""
    return label, font, set(indices)


# ---------------------------------------------------------------------------
# Arrow endpoint → shape matching
# ---------------------------------------------------------------------------

def _dist_to_bbox(x, y, b, margin=10):
    """distance from point (x,y) to bbox b (returns 0 if inside the expanded box)."""
    xn, yn = b[0] - margin, b[1] - margin
    xx, yx = b[2] + margin, b[3] + margin
    if xn <= x <= xx and yn <= y <= yx:
        return 0.0
    return math.sqrt(max(xn - x, 0, x - xx) ** 2 + max(yn - y, 0, y - yx) ** 2)


def _find_shape_for_point(x, y, shape_entities, max_dist=150):
    """return the id of the nearest shape to (x, y), or None if nothing within max_dist."""
    best_k, best_d = None, max_dist
    for k, v in shape_entities.items():
        if "bounds" not in v:   # skip arrow entities mixed into the dict
            continue
        d = _dist_to_bbox(x, y, v["bounds"])
        if d < best_d:
            best_d, best_k = d, k
    return best_k


# ---------------------------------------------------------------------------
# Core reconstruction  (GT SVG → entities dict)
# ---------------------------------------------------------------------------

def _collect_shape_elems(root):
    """find all elements with id='shape_N' and return them sorted by N.

    returns list of (int_index, idx_str, elem, tag).

    stacked GT shapes assign id='shape_N' to every layer (outer <g>, shadow
    copies, inner face).  we keep only the FIRST element found per id (tree
    order = outermost), which is the canonical shape container.
    """
    seen = set()
    shape_elems = []
    for elem in root.iter():
        eid = elem.get("id", "")
        if not eid.startswith("shape_"):
            continue
        idx_str = eid[len("shape_"):]
        if not idx_str.isdigit():
            continue
        if eid in seen:
            continue  # skip shadow/inner copies that share the same id
        seen.add(eid)
        shape_elems.append((int(idx_str), idx_str, elem, strip_ns(elem.tag)))
    shape_elems.sort(key=lambda x: x[0])
    return shape_elems


def _collect_shape_elems_fallback(root):
    """fallback when the svg has no shape_N ids.

    scans elements for diagram shapes, recursing into <g transform> containers.
    assigns ids shape_0, shape_1, ... by document order.
    handles <use href="#id"> elements by resolving against <defs>.
    """
    # parse CSS so we can resolve class-based fills
    from eval_shapes import (parse_css, css_get as _css_get,
                             _build_defs_map, _bbox_use, _bbox_group_resolve_uses,
                             _subtree_has_shape_or_use)
    css = parse_css(root)
    defs_map = _build_defs_map(root)

    vb = root.get("viewBox", "")
    vb_nums = find_nums(vb)
    canvas_w = vb_nums[2] if vb_nums and len(vb_nums) >= 4 else safe_float(root.get("width", "800"))
    canvas_h = vb_nums[3] if vb_nums and len(vb_nums) >= 4 else safe_float(root.get("height", "600"))

    skip_tags = {"defs", "style", "title", "desc", "marker", "filter",
                 "linearGradient", "radialGradient", "pattern", "symbol"}

    shape_elems = []
    idx_counter = [0]

    def _collect(elems, ox=0.0, oy=0.0):
        for elem in elems:
            tag = strip_ns(elem.tag)
            if tag in skip_tags:
                continue

            if tag in SHAPE_TAGS:
                attrs = elem.attrib
                # resolve fill via inline attr first, then CSS class
                fill = attrs.get("fill", "").strip().lower()
                if not fill:
                    fill = _css_get(elem, "fill", css).strip().lower()

                # skip explicit fill=none (borders/containers)
                if fill == "none":
                    continue

                if tag == "rect":
                    w_str = attrs.get("width", "0")
                    h_str = attrs.get("height", "0")
                    if "%" in w_str or "%" in h_str:
                        continue
                    w = safe_float(w_str)
                    h = safe_float(h_str)
                    # skip full-canvas background rect
                    if abs(w - canvas_w) < 5 and abs(h - canvas_h) < 5:
                        continue
                    # skip rects covering >50% of canvas (section containers)
                    if canvas_w > 0 and canvas_h > 0:
                        if (w * h) / (canvas_w * canvas_h) > 0.5:
                            continue

                # wrap element in a small proxy that shifts its coords by (ox, oy)
                # by storing the offset on it — _shape_bounds uses bbox() which
                # reads inline attrs, so we patch the element temporarily
                idx_str = str(idx_counter[0])
                idx_counter[0] += 1

                if ox != 0.0 or oy != 0.0:
                    # create a lightweight wrapper preserving the offset
                    import copy
                    proxy = copy.copy(elem)
                    proxy.attrib = dict(elem.attrib)
                    # shift positional attrs
                    for attr in ("x", "cx"):
                        if attr in proxy.attrib:
                            proxy.attrib[attr] = str(safe_float(proxy.attrib[attr]) + ox)
                    for attr in ("y", "cy"):
                        if attr in proxy.attrib:
                            proxy.attrib[attr] = str(safe_float(proxy.attrib[attr]) + oy)
                    shape_elems.append((int(idx_str), idx_str, proxy, tag))
                else:
                    shape_elems.append((int(idx_str), idx_str, elem, tag))
                continue

            if tag == "use":
                # Resolve <use> directly as a shape
                b = _bbox_use(elem, defs_map, ox, oy)
                if b and (b[2]-b[0]) >= 15 and (b[3]-b[1]) >= 15:
                    idx_str = str(idx_counter[0])
                    idx_counter[0] += 1
                    shape_elems.append((int(idx_str), idx_str, elem, tag))
                continue

            if tag == "g":
                # accumulate translate transform
                tx, ty = parse_translate(elem.get("transform", ""))
                new_ox, new_oy = ox + tx, oy + ty

                ctags = {strip_ns(c.tag) for c in elem}
                is_arrow = (ctags & {"path", "line"}) and "polygon" in ctags
                has_text  = "text" in ctags
                has_shape = bool(ctags & SHAPE_TAGS)
                # also consider <use> children and nested groups with uses/shapes
                has_use = "use" in ctags
                has_shape_or_use = has_shape or has_use or any(
                    _subtree_has_shape_or_use(c)
                    for c in elem if strip_ns(c.tag) == "g"
                )

                # if this <g> contains multiple distinct shapes, recurse into it
                shape_children = [c for c in elem if strip_ns(c.tag) in SHAPE_TAGS]
                use_children   = [c for c in elem if strip_ns(c.tag) == "use"]
                # a <g> with direct <use> children AND a text label is a single
                # composite shape (e.g. use-based scenario element with caption) —
                # don't let the transform attr promote it to a container
                is_container = (
                    len(shape_children) + len(use_children) > 2
                    or "transform" in elem.attrib
                ) and not (has_use and has_text)

                if is_container and has_shape_or_use:
                    _collect(elem, new_ox, new_oy)
                elif has_shape_or_use and (has_text or not is_arrow):
                    # treat as a single composite shape (e.g. 3D prism, use-based scenario)
                    idx_str = str(idx_counter[0])
                    idx_counter[0] += 1
                    shape_elems.append((int(idx_str), idx_str, elem, tag))

    _collect(root)
    shape_elems.sort(key=lambda x: x[0])
    return shape_elems


def _collect_arrow_groups(root, svg_text):
    """Extract arrow elements from the svg.

    Strategy A: look for the <!-- Arrows --> ... <!-- Shapes --> comment block
    (eval-set gt format) and parse only that section.

    Strategy B: scan root-level <g> elements that contain both a shaft
    (line or path) and an inline polygon arrowhead as direct siblings.

    Strategy C (fallback): collect any standalone <line>/<path> with a
    marker-end attribute anywhere in the document.  Each is returned as the
    element itself (not wrapped in a <g>).
    """
    arrow_groups = []

    # Strategy A: comment-bounded section (eval-set format)
    m = re.search(r"<!--\s*Arrows.*?-->(.*?)<!--\s*Shapes", svg_text, re.DOTALL)
    if m:
        section = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", m.group(1))
        try:
            arr_root = ET.fromstring(f"<root>{section}</root>")
            for g in arr_root:
                if g.tag != "g":
                    continue
                ctags = {c.tag.split("}")[-1] for c in g}
                if ctags & {"path", "line"} and "polygon" in ctags:
                    arrow_groups.append(g)
        except ET.ParseError:
            pass

    # Strategy B: root-level <g> with inline polygon arrowhead
    if not arrow_groups:
        for elem in root:
            tag = strip_ns(elem.tag)
            if tag != "g" or elem.get("id", "").startswith("shape_"):
                continue
            ctags = {strip_ns(c.tag) for c in elem}
            if ctags & {"path", "line"} and "polygon" in ctags:
                arrow_groups.append(elem)

    # Strategy C: standalone marker-end lines/paths (common in author-SVGs)
    if not arrow_groups:
        seen = set()
        for elem in root.iter():
            tag = strip_ns(elem.tag)
            if tag not in ("line", "path"):
                continue
            me = (elem.get("marker-end") or elem.get("marker-start") or "").strip()
            if not me or me.lower() == "none":
                continue
            eid = id(elem)
            if eid in seen:
                continue
            seen.add(eid)
            arrow_groups.append(elem)  # standalone element, not a group

    return arrow_groups


def reconstruct_meta(gt_svg_path):
    """parse a gt svg and return a metadata dict matching the json schema.

    reconstructed fields per shape:
      id, type, bounds, label, font, fillStyle, fillColor,
      strokeColor, strokeWidth, borderStyle, stacked

    reconstructed fields per arrow:
      id, type, from, to, color, width, style, dash
    """
    gt_svg_path = Path(gt_svg_path)
    tree        = ET.parse(str(gt_svg_path))
    root        = tree.getroot()
    svg_text    = gt_svg_path.read_text(encoding="utf-8")

    defs         = parse_defs(root)
    fill_clr_map = _build_fill_color_map(root)
    all_texts    = _collect_texts(root)
    used_texts   = set()
    entities     = {}

    from eval_shapes import _build_defs_map as _bdm
    _defs_map = _bdm(root)

    # ------------------------------------------------------------------ shapes

    shape_elems = _collect_shape_elems(root)
    if not shape_elems:
        # no shape_N ids found — try document-order fallback
        shape_elems = _collect_shape_elems_fallback(root)

    for _, idx_str, elem, tag in shape_elems:
        key = f"shape_{idx_str}"
        b   = _shape_bounds(elem, tag, defs_map=_defs_map)
        # Skip degenerate shapes (thin strokes, individual dashes, etc.)
        if (b[2] - b[0]) < 5 or (b[3] - b[1]) < 5:
            continue

        # get the primary element for reading visual attributes
        if tag == "use":
            href = elem.get("href") or elem.get("{http://www.w3.org/1999/xlink}href", "")
            _ref = _defs_map.get(href.lstrip("#"))
            main_elem = _ref if _ref is not None else elem
            main_tag  = strip_ns(main_elem.tag) if _ref is not None else "use"
        elif tag == "g":
            main_elem, main_tag = _main_child(elem)
        else:
            main_elem, main_tag = elem, tag

        # fill color: resolve url(#...) via the defs color map
        raw_fill   = main_elem.get("fill", "") or elem.get("fill", "")
        fill_color = _resolve_fill_color(raw_fill, fill_clr_map)

        # fill style: check main element first, then hunt through descendants
        # (some shapes apply a pattern overlay on a second child element)
        fill_style = detect_fill_style(main_elem, defs)
        if fill_style in ("solid", "none"):
            for c in elem.iter():
                if strip_ns(c.tag) in SHAPE_TAGS:
                    fs = detect_fill_style(c, defs)
                    if fs not in ("solid", "none"):
                        fill_style = fs
                        fc = _resolve_fill_color(c.get("fill", ""), fill_clr_map)
                        if fc and not fc.startswith("url("):
                            fill_color = fc
                        break

        stroke_color = main_elem.get("stroke", "") or elem.get("stroke", "")
        stroke_width = safe_float(
            main_elem.get("stroke-width", elem.get("stroke-width", "2")))
        border_style = detect_border_style(main_elem)
        shape_type   = infer_shape_type(elem, tag)
        stacked      = _count_stacked_layers(elem) + 1  # 1 = no depth layers

        label, font, consumed = _collect_label_texts(b, all_texts, used_texts)
        if not label:
            label = key  # fallback: use the shape id as label
        used_texts |= consumed

        entities[key] = {
            "id":          key,
            "type":        shape_type,
            "bounds":      [round(b[0]), round(b[1]), round(b[2]), round(b[3])],
            "label":       label,
            "font":        font,
            "fillStyle":   fill_style,
            "fillColor":   fill_color,
            "strokeColor": stroke_color,
            "strokeWidth": stroke_width,
            "borderStyle": border_style,
            "stacked":     stacked,
        }

    # ------------------------------------------------------------------ arrows

    arrow_groups = _collect_arrow_groups(root, svg_text)

    for i, g in enumerate(arrow_groups):
        key      = f"arrow_{i}"
        color    = "#000000"
        width    = 2.0
        style    = "straight"
        dash     = "none"
        start_pt = end_pt = None

        # g may be a <g> container OR a standalone <line>/<path> (Strategy C)
        g_tag = strip_ns(g.tag)
        shaft_elems = list(g) if g_tag == "g" else [g]

        for c in shaft_elems:
            ctag = c.tag.split("}")[-1] if "}" in c.tag else c.tag  # strip ns

            if ctag == "line":
                color    = c.get("stroke", color)
                width    = safe_float(c.get("stroke-width") or str(width))
                da       = c.get("stroke-dasharray", "")
                dash     = da if da else "none"
                start_pt = (safe_float(c.get("x1", 0)), safe_float(c.get("y1", 0)))
                end_pt   = (safe_float(c.get("x2", 0)), safe_float(c.get("y2", 0)))
                style    = "straight"

            elif ctag == "path":
                color    = c.get("stroke", color)
                width    = safe_float(c.get("stroke-width") or str(width))
                da       = c.get("stroke-dasharray", "")
                dash     = da if da else "none"
                d        = c.get("d", "")
                start_pt = path_start(d)
                end_pt   = path_end(d)
                style    = "curved" if detect_curvature(d)["curved"] else "straight"

        from_key = _find_shape_for_point(*start_pt, entities) if start_pt else None
        to_key   = _find_shape_for_point(*end_pt,   entities) if end_pt   else None

        # avoid self-loops: if both ends hit the same shape, re-match the endpoint
        if from_key == to_key and from_key is not None and end_pt:
            fallback = {k: v for k, v in entities.items() if k != from_key}
            to_key   = _find_shape_for_point(*end_pt, fallback) or to_key

        entities[key] = {
            "id":    key,
            "type":  "arrow",
            "from":  from_key or "shape_0",
            "to":    to_key   or "shape_0",
            "color": color,
            "width": width,
            "style": style,
            "dash":  dash,
        }

    return {"entities": entities}


# ---------------------------------------------------------------------------
# Evaluation wrapper
# ---------------------------------------------------------------------------

def evaluate_standalone(gen_svg, gt_svg, ann_dir=None, label=None):
    """evaluate gen_svg against gt_svg, reconstructing metadata from the gt svg.

    returns (shape_scores, shape_report, arrow_scores, arrow_report).
    """
    gen_svg = Path(gen_svg)
    gt_svg  = Path(gt_svg)
    sid     = label or gen_svg.stem

    meta = reconstruct_meta(gt_svg)

    # write reconstructed metadata to a temp json so the evaluators can read it
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(meta, tmp, indent=2)
    tmp.close()
    tmp_json = Path(tmp.name)

    try:
        ann_shapes = Path(ann_dir) / f"{sid}_shapes.svg" if ann_dir else None
        ann_arrows = Path(ann_dir) / f"{sid}_arrows.svg" if ann_dir else None

        shape_scores, shape_report = evaluate_shapes_file(
            gen_svg, gt_svg, tmp_json, ann_shapes, sid)
        arrow_scores, arrow_report = evaluate_arrows_file(
            gen_svg, gt_svg, tmp_json, ann_arrows, sid)
    finally:
        tmp_json.unlink(missing_ok=True)

    return shape_scores, shape_report, arrow_scores, arrow_report


def write_gt_annotation(gt_svg, meta, out_path):
    """Draw reconstruct_meta detections on the GT SVG and save to out_path.

    Each detected shape is drawn as a green dashed box with its label,
    so you can verify the GT reconstruction quality independently.
    """
    svg_text = Path(gt_svg).read_text(encoding="utf-8")

    boxes = []
    for key, ent in meta.get("entities", {}).items():
        if ent.get("type") == "arrow":
            continue
        b = ent.get("bounds", [0, 0, 0, 0])
        x, y, w, h = b[0], b[1], b[2] - b[0], b[3] - b[1]
        label = ent.get("label", key)
        boxes.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="rgba(0,200,0,0.08)" stroke="#00CC00" stroke-width="2" '
            f'stroke-dasharray="8,4" />'
            f'<text x="{x+3}" y="{max(y-3,8)}" font-family="monospace" font-size="10" '
            f'fill="#008800" font-weight="bold">{key}: {label[:40]}</text>'
        )

    overlay = '<g id="gt-reconstruction-overlay" opacity="0.9">' + "".join(boxes) + "</g>"
    # insert before closing </svg>
    annotated = svg_text.rstrip()
    if annotated.endswith("</svg>"):
        annotated = annotated[:-6] + overlay + "\n</svg>"
    else:
        annotated += overlay

    Path(out_path).write_text(annotated, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_model_dir(name):
    """find the model folder: check input/<name> first, then the script root."""
    p = INPUT_DIR / name
    if p.exists():
        return p
    p = BASE / name
    if p.exists():
        return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model folder without a GT JSON file.")
    parser.add_argument("input_folder", nargs="?",
                        help="model folder name (checked in input/ first). "
                             "omit with -d for dry run.")
    parser.add_argument("-t", action="store_true",
                        help="write annotation svgs to output/<model>/")
    parser.add_argument("-d", action="store_true",
                        help="dry run: use gt svg as model input (should score ~1.0)")
    parser.add_argument("-v", action="store_true",
                        help="verbose: per-sample scores")
    parser.add_argument("--gt-svg", default=None,
                        help="path to gt svg folder (auto-detected if omitted)")
    parser.add_argument("--input-dir", default=None,
                        help="folder containing model SVGs (overrides input/<model>)")
    parser.add_argument("--sample", default=None,
                        help="evaluate only this sample ID (GT SVG stem), for debugging")
    args = parser.parse_args()

    # resolve gt svg dir
    gt_svg_dir = Path(args.gt_svg) if args.gt_svg else _find_gt_svg_dir()
    if not gt_svg_dir or not gt_svg_dir.exists():
        print("ERROR: GT SVG folder not found. Expected: eval-set/ (flat folder of *.svg files)")
        print("Use --gt-svg to specify the folder explicitly.")
        sys.exit(1)

    # resolve model dir
    if args.d:
        model_dir  = gt_svg_dir
        model_name = "dry-run"
    else:
        if not args.input_folder:
            parser.print_help(); sys.exit(1)
        if args.input_dir:
            model_dir = Path(args.input_dir) / args.input_folder
        else:
            model_dir = _resolve_model_dir(args.input_folder)
        if model_dir is None or not model_dir.exists():
            print(f"ERROR: '{args.input_folder}' not found.")
            sys.exit(1)
        model_name = model_dir.name

    ann_dir = OUTPUT_DIR / model_name if args.t else None
    if ann_dir:
        if ann_dir.exists():
            shutil.rmtree(ann_dir)
        ann_dir.mkdir(parents=True)

    out_dir = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # All GT samples; match model files by exact stem or prefix
    # (handles both numeric 000000.svg and named arxiv_svg__foo_model.svg)
    gt_all = {f.stem: f for f in sorted(gt_svg_dir.glob("*.svg"))}
    if args.sample:
        gt_all = {k: v for k, v in gt_all.items() if args.sample in k}
    all_model = {f.stem: f for f in model_dir.glob("*.svg")}
    model_files = {}
    for sid in gt_all:
        if sid in all_model:
            model_files[sid] = all_model[sid]
            continue
        # numeric zero-padding fallback (000000 <-> 0)
        m = re.match(r"^0*(\d+)$", sid)
        if m and m.group(1).zfill(6) in all_model:
            model_files[sid] = all_model[m.group(1).zfill(6)]
            continue
        # prefix match: model stem starts with GT stem + separator
        for stem, f in all_model.items():
            if stem.startswith(sid + "_") or stem.startswith(sid + "-"):
                model_files[sid] = f
                break

    if not gt_all:
        print("No GT samples found."); sys.exit(1)

    if args.v:
        print(f"GT SVGs  : {gt_svg_dir}  ({len(gt_all)} samples)")
        print(f"Input    : {model_dir}  ({len(all_model)} SVGs, {len(model_files)} matched)")
        if ann_dir:
            print(f"Annots   : {ann_dir}/")
        print()

    shape_totals = defaultdict(list)
    arrow_totals = defaultdict(list)
    missing, errored = [], {}

    for sid, gt_svg in sorted(gt_all.items()):
        gen_svg = model_files.get(sid)

        if gen_svg is None:
            # model produced no output for this sample → 0.0 on everything
            missing.append(sid)
            for k, v in ZERO_SHAPE_SCORES.items(): shape_totals[k].append(v)
            for k, v in ZERO_ARROW_SCORES.items(): arrow_totals[k].append(v)
            if args.v:
                print(f"  {sid} ... MISSING")
            continue

        if args.v:
            print(f"  {sid} ... ", end="", flush=True)

        try:
            s_sc, s_rep, a_sc, a_rep = evaluate_standalone(
                gen_svg, gt_svg, ann_dir=ann_dir, label=sid)
            (out_dir / f"{sid}_shapes.txt").write_text(s_rep, encoding="utf-8")
            (out_dir / f"{sid}_arrows.txt").write_text(a_rep, encoding="utf-8")
            if ann_dir:
                meta = reconstruct_meta(gt_svg)
                write_gt_annotation(gt_svg, meta, out_dir / f"{sid}_gt.svg")
            for k, v in s_sc.items(): shape_totals[k].append(v)
            for k, v in a_sc.items(): arrow_totals[k].append(v)
            if args.v:
                print(f"shapes={s_sc['composite']:.3f}  arrows={a_sc['composite']:.3f}")
        except Exception as e:
            errored[sid] = str(e)
            for k, v in ZERO_SHAPE_SCORES.items(): shape_totals[k].append(v)
            for k, v in ZERO_ARROW_SCORES.items(): arrow_totals[k].append(v)
            if args.v:
                print(f"ERROR: {e}")

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    print(f"\n{'='*55}")
    print("SHAPE EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Metric':<15} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*43}")
    for c in ["label", "type", "fill_color", "fill_style", "stroke_color",
              "border_style", "position", "font", "aspect_ratio",
              "extra_penalty", "composite"]:
        v = shape_totals.get(c, [])
        if v:
            print(f"  {c:<15} {avg(v):>8.3f} {min(v):>8.3f} {max(v):>8.3f}")

    print(f"\n{'='*55}")
    print("ARROW EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Metric':<15} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*43}")
    for c in ["source", "dest", "head", "head_size", "curve", "color",
              "overlap", "extra_penalty", "composite"]:
        v = arrow_totals.get(c, [])
        if v:
            print(f"  {c:<15} {avg(v):>8.3f} {min(v):>8.3f} {max(v):>8.3f}")

    sc = shape_totals.get("composite", [])
    ac = arrow_totals.get("composite", [])
    if sc and ac:
        print(f"\n{'='*55}")
        print(f"OVERALL: {(avg(sc) + avg(ac)) / 2:.3f}  (50% shapes + 50% arrows)")
        print(f"{'='*55}")

    n_total = len(gt_all)
    print(f"\nMissing : {len(missing)}/{n_total}")
    print(f"Errored : {len(errored)}/{n_total}")
    if errored:
        (OUTPUT_DIR / model_name / "errors.txt").write_text(
            "\n".join(f"{sid}: {e}" for sid, e in sorted(errored.items())),
            encoding="utf-8")

    print(f"Outputs : {OUTPUT_DIR / model_name}/")


if __name__ == "__main__":
    main()
