#!/usr/bin/env python3
"""Shape evaluation module - scores generated SVG shapes against ground truth.

Can be run standalone to evaluate a single SVG pair:
    python lib/eval_shapes.py generated.svg gt.svg metadata.json
    python lib/eval_shapes.py generated.svg gt.svg metadata.json --annotate out.svg
"""

import sys, json, math, re
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import strip_ns, safe_float, find_nums, parse_translate, normalize_label, hex_to_rgb, color_score, canvas_size

# Helpers

def font_score(f1, f2):
    n1, n2 = normalize_label(f1).split(",")[0], normalize_label(f2).split(",")[0]
    if not n1 or not n2: return 0.0
    if n1 == n2: return 1.0
    serif = {"times", "georgia", "palatino", "garamond"}
    sans = {"arial", "helvetica", "verdana", "tahoma", "trebuchet"}
    mono = {"courier", "consolas", "monaco"}
    def fam(f):
        if any(x in f for x in serif): return "serif"
        if any(x in f for x in sans): return "sans"
        if any(x in f for x in mono): return "mono"
        return None
    return 0.5 if fam(n1) and fam(n1) == fam(n2) else 0.0

def collect_text(elem):
    parts = [elem.text.strip()] if elem.text and elem.text.strip() else []
    for child in elem:
        parts.append(collect_text(child))
        if child.tail and child.tail.strip(): parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)

def parse_css(root):
    """Parse <style> block into {class_name: {prop: value}} dict.

    Handles both plain CSS and CDATA-wrapped blocks.  Only class selectors
    (.foo { ... }) are extracted; tag/id selectors are ignored.
    """
    css = {}
    for elem in root.iter():
        if strip_ns(elem.tag) != "style":
            continue
        text = elem.text or ""
        text = re.sub(r"<!\[CDATA\[|\]\]>", "", text)
        for m in re.finditer(r"\.([^{,\s]+)\s*\{([^}]*)\}", text):
            cls, body = m.group(1), m.group(2)
            props = {}
            for decl in body.split(";"):
                if ":" in decl:
                    k, v = decl.split(":", 1)
                    props[k.strip()] = v.strip()
            if props:
                css[cls] = props
    return css


def css_get(elem, prop, css):
    """Read a CSS property from an element, with fallback priority:
    1. inline attribute  (e.g. fill="#abc")
    2. inline style=""   (e.g. style="fill:#abc")
    3. class rule        (e.g. class="node" with .node{fill:#abc} in <style>)
    """
    val = elem.get(prop, "")
    if val:
        return val
    style_attr = elem.get("style", "")
    if style_attr:
        for decl in style_attr.split(";"):
            if ":" in decl:
                k, v = decl.split(":", 1)
                if k.strip() == prop:
                    return v.strip()
    if css:
        for cls in (elem.get("class") or "").split():
            if cls in css and prop in css[cls]:
                return css[cls][prop]
    return ""


def get_font(elem, parent_map=None, css=None):
    f = css_get(elem, "font-family", css or {})
    if f: return f
    for c in elem:
        if strip_ns(c.tag) == "tspan":
            f = css_get(c, "font-family", css or {})
            if f: return f
    if parent_map:
        cur = elem
        while cur in parent_map:
            cur = parent_map[cur]
            f = css_get(cur, "font-family", css or {})
            if f: return f
    return ""

# Bounding box

SHAPE_TAGS = {"rect", "circle", "ellipse", "polygon", "polyline", "path"}


def _build_defs_map(root):
    """Build {id: element} from all <defs> children of the SVG root."""
    defs_map = {}
    for child in root:
        if strip_ns(child.tag) == "defs":
            for elem in child.iter():
                eid = elem.get("id")
                if eid:
                    defs_map[eid] = elem
    return defs_map


def _bbox_from_def(ref, defs_map, ox=0.0, oy=0.0):
    """Recursively compute bbox of a defs element, resolving nested <use>."""
    tag = strip_ns(ref.tag)
    if tag in SHAPE_TAGS:
        return bbox(ref, tag, ox, oy)
    if tag == "use":
        href = ref.get("href") or ref.get("{http://www.w3.org/1999/xlink}href", "")
        nested = defs_map.get(href.lstrip("#"))
        if nested is None:
            return None
        return _bbox_from_def(nested, defs_map, ox + safe_float(ref.get("x", 0)),
                              oy + safe_float(ref.get("y", 0)))
    if tag == "g":
        tx, ty = parse_translate(ref.get("transform", ""))
        merged = None
        for child in ref:
            b = _bbox_from_def(child, defs_map, ox + tx, oy + ty)
            if b:
                merged = b if merged is None else [
                    min(merged[0], b[0]), min(merged[1], b[1]),
                    max(merged[2], b[2]), max(merged[3], b[3])]
        return merged
    return None


def _bbox_use(use_elem, defs_map, ox=0.0, oy=0.0):
    """Compute bbox for a <use> element by resolving its href."""
    href = use_elem.get("href") or use_elem.get("{http://www.w3.org/1999/xlink}href", "")
    ref = defs_map.get(href.lstrip("#"))
    if ref is None:
        return None
    return _bbox_from_def(ref, defs_map, ox + safe_float(use_elem.get("x", 0)),
                          oy + safe_float(use_elem.get("y", 0)))


def _subtree_has_shape_or_use(elem):
    """True if any descendant of elem is a shape or <use> element."""
    for child in elem:
        ctag = strip_ns(child.tag)
        if ctag in SHAPE_TAGS or ctag == "use":
            return True
        if ctag == "g" and _subtree_has_shape_or_use(child):
            return True
    return False


def _bbox_group_resolve_uses(elem, defs_map, ox=0.0, oy=0.0):
    """Union bbox of all shapes and resolved <use> elements within a group."""
    tx, ty = parse_translate(elem.get("transform", ""))
    gx, gy = ox + tx, oy + ty
    merged = None
    for child in elem:
        ctag = strip_ns(child.tag)
        if ctag in SHAPE_TAGS:
            b = bbox(child, ctag, gx, gy)
        elif ctag == "use":
            b = _bbox_use(child, defs_map, gx, gy)
        elif ctag == "g":
            b = _bbox_group_resolve_uses(child, defs_map, gx, gy)
        else:
            continue
        if b:
            merged = b if merged is None else [
                min(merged[0], b[0]), min(merged[1], b[1]),
                max(merged[2], b[2]), max(merged[3], b[3])]
    return merged


def bbox(elem, tag, ox=0.0, oy=0.0):
    if tag == "rect":
        x, y = safe_float(elem.get("x")), safe_float(elem.get("y"))
        w, h = safe_float(elem.get("width")), safe_float(elem.get("height"))
        return [x+ox, y+oy, x+w+ox, y+h+oy]
    if tag == "circle":
        cx, cy, r = safe_float(elem.get("cx")), safe_float(elem.get("cy")), safe_float(elem.get("r"))
        return [cx-r+ox, cy-r+oy, cx+r+ox, cy+r+oy]
    if tag == "ellipse":
        cx, cy = safe_float(elem.get("cx")), safe_float(elem.get("cy"))
        rx, ry = safe_float(elem.get("rx")), safe_float(elem.get("ry"))
        return [cx-rx+ox, cy-ry+oy, cx+rx+ox, cy+ry+oy]
    if tag in ("polygon", "polyline"):
        nums = find_nums(elem.get("points", ""))
        if len(nums) < 4: return None
        xs, ys = [nums[i]+ox for i in range(0, len(nums), 2)], [nums[i]+oy for i in range(1, len(nums), 2)]
        return [min(xs), min(ys), max(xs), max(ys)]
    if tag == "path":
        coords = parse_path_coords(elem.get("d", ""))
        if len(coords) < 2: return None
        xs, ys = [p[0]+ox for p in coords], [p[1]+oy for p in coords]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None

def parse_path_coords(d):
    """Parse SVG path data into a list of (x, y) endpoint coordinates.

    Tracks current position so relative commands (lowercase) produce correct
    absolute coordinates.  Bezier control points are intentionally omitted —
    only curve endpoints are recorded — so the resulting bbox reflects the
    actual visible extent of the path rather than the (larger) convex hull of
    control points.
    """
    coords, tokens = [], re.findall(r"[MmLlHhVvQqCcSsTtAaZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    i, cmd = 0, "M"
    cx, cy = 0.0, 0.0   # current pen position
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd, i = tokens[i], i + 1
            continue
        try:
            uc = cmd.upper()
            rel = cmd.islower()   # relative vs absolute

            if uc in "MLT" and i + 1 < len(tokens):
                x, y = float(tokens[i]), float(tokens[i + 1])
                if rel: x, y = cx + x, cy + y
                coords.append((x, y))
                cx, cy = x, y
                i += 2

            elif uc == "H" and i < len(tokens):
                x = float(tokens[i])
                if rel: x = cx + x
                coords.append((x, cy))
                cx = x
                i += 1

            elif uc == "V" and i < len(tokens):
                y = float(tokens[i])
                if rel: y = cy + y
                coords.append((cx, y))
                cy = y
                i += 1

            elif uc == "Q" and i + 3 < len(tokens):
                # quadratic bezier — record endpoint only
                x, y = float(tokens[i + 2]), float(tokens[i + 3])
                if rel: x, y = cx + x, cy + y
                coords.append((x, y))
                cx, cy = x, y
                i += 4

            elif uc == "C" and i + 5 < len(tokens):
                # cubic bezier — record endpoint only (skip control points)
                x, y = float(tokens[i + 4]), float(tokens[i + 5])
                if rel: x, y = cx + x, cy + y
                coords.append((x, y))
                cx, cy = x, y
                i += 6

            elif uc == "S" and i + 3 < len(tokens):
                # smooth cubic — record endpoint only
                x, y = float(tokens[i + 2]), float(tokens[i + 3])
                if rel: x, y = cx + x, cy + y
                coords.append((x, y))
                cx, cy = x, y
                i += 4

            elif uc == "A" and i + 6 < len(tokens):
                # arc — record endpoint
                x, y = float(tokens[i + 5]), float(tokens[i + 6])
                if rel: x, y = cx + x, cy + y
                coords.append((x, y))
                cx, cy = x, y
                i += 7

            else:
                i += 1
        except Exception:
            i += 1
    return coords

def center(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def merge_bbox(a, b):
    if not a: return b
    if not b: return a
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

def _overlaps(a, b):
    """Check if two bounding boxes overlap significantly."""
    if not a or not b: return False
    m = 15
    return not (a[2]+m < b[0] or b[2]+m < a[0] or a[3]+m < b[1] or b[3]+m < a[1])

def is_shape(elem, tag, b, css=None, canvas_wh=None):
    if not b: return False
    w, h = b[2]-b[0], b[3]-b[1]
    if w < 15 or h < 15: return False
    if tag == "line": return False
    fill   = css_get(elem, "fill",   css or {}).strip().lower()
    stroke = css_get(elem, "stroke", css or {}).strip().lower()
    # skip unfilled elements (borders/containers/arrows)
    if tag in ("rect", "circle", "ellipse", "polygon", "polyline"):
        if not fill or fill == "none":
            # rect/circle/ellipse/polygon with no fill but a visible stroke are
            # container shapes (e.g. .container-box { fill:none; stroke:#D50000 })
            has_stroke = bool(stroke) and stroke != "none"
            if not has_stroke or tag == "polyline":
                return False
    if tag == "path":
        # paths with fill="none" are typically connectors/arrows — skip
        if not fill or fill == "none":
            return False
    # skip rects covering more than 50% of canvas (section containers)
    if tag == "rect" and canvas_wh:
        cw, ch = canvas_wh
        if cw > 0 and ch > 0 and (w * h) / (cw * ch) > 0.5:
            return False
    return True

# Fill and stroke style detection

def parse_defs(root):
    defs = {}
    for elem in root.iter():
        if strip_ns(elem.tag) == "defs":
            for child in elem:
                tag, eid = strip_ns(child.tag), child.get("id", "")
                if tag == "pattern":
                    has_circle = any(strip_ns(c.tag) == "circle" for c in child)
                    has_line = any(strip_ns(c.tag) == "line" for c in child)
                    has_path = any(strip_ns(c.tag) == "path" for c in child)
                    lines = [c for c in child if strip_ns(c.tag) == "line"]
                    if has_circle: defs[eid] = "dots"
                    elif len(lines) >= 2: defs[eid] = "crosshatch"
                    elif has_line:
                        if lines and abs(safe_float(lines[0].get("y1")) - safe_float(lines[0].get("y2"))) > 1:
                            defs[eid] = "hatching"
                        else: defs[eid] = "horizontal-lines"
                    elif has_path and not has_circle and not has_line:
                        # Detect hatching/crosshatch encoded as path segments
                        paths = [c for c in child if strip_ns(c.tag) == "path"]
                        seg_angles = set()
                        for p in paths:
                            pd = p.get("d", "")
                            # Extract line-like segments (M...L pairs)
                            segs = re.findall(r"[Mm]\s*([-\d.]+)[,\s]+([-\d.]+).*?[Ll]\s*([-\d.]+)[,\s]+([-\d.]+)", pd)
                            for s in segs:
                                dx = float(s[2]) - float(s[0])
                                dy = float(s[3]) - float(s[1])
                                if abs(dx) > 0.1 or abs(dy) > 0.1:
                                    angle = round(math.atan2(dy, dx) * 180 / math.pi / 15) * 15
                                    seg_angles.add(angle % 180)
                        if len(seg_angles) >= 2:
                            defs[eid] = "crosshatch"
                        elif len(seg_angles) == 1:
                            defs[eid] = "hatching"
                        else:
                            defs[eid] = "pattern"
                    else: defs[eid] = "pattern"
                elif tag == "linearGradient": defs[eid] = "gradient-linear"
                elif tag == "radialGradient": defs[eid] = "gradient-radial"
    return defs

def parse_gradient_colors(root):
    """Return {gradient_id: first_stop_hex} for all linear/radial gradients in the SVG."""
    colors = {}
    for elem in root.iter():
        if strip_ns(elem.tag) not in ("linearGradient", "radialGradient"): continue
        eid = elem.get("id", "")
        if not eid: continue
        for stop in elem:
            if strip_ns(stop.tag) != "stop": continue
            sc = stop.get("stop-color", "")
            if not sc:
                style = stop.get("style", "")
                m = re.search(r"stop-color\s*:\s*([^;]+)", style)
                sc = m.group(1).strip() if m else ""
            if sc and sc.lower() not in ("none", ""):
                colors[eid] = sc
                break
    return colors

def resolve_fill_color(fill, grad_colors):
    """Resolve a fill attribute to a hex color string, or return the original."""
    if fill.startswith("url(#"):
        ref = fill[5:-1]
        return grad_colors.get(ref, fill)
    return fill

def detect_fill_style(elem, defs, css=None):
    fill = css_get(elem, "fill", css or {})
    if fill.startswith("url(#"):
        ref = fill[5:-1]
        return defs.get(ref, "pattern")
    if fill and fill.lower() != "none":
        return "solid"
    return "none"

def detect_border_style(elem, css=None):
    da = css_get(elem, "stroke-dasharray", css or {})
    if not da: return "solid"
    nums = find_nums(da)
    if not nums: return "solid"
    if len(nums) >= 4: return "dash-dot"
    if nums[0] <= 3: return "dotted"
    return "dashed"

def detect_rounded(elem, tag):
    if tag == "rect":
        rx, ry = safe_float(elem.get("rx")), safe_float(elem.get("ry"))
        return rx > 0 or ry > 0
    if tag == "polygon":
        return elem.get("stroke-linejoin", "").lower() == "round"
    return False

def fill_style_score(expected, detected):
    if expected == detected: return 1.0
    # Both non-solid but different pattern type (e.g. crosshatch vs dots)
    if expected != "solid" and detected != "solid": return 0.5
    # One solid, one patterned — completely wrong category
    return 0.0

def border_style_score(expected, detected):
    if expected == detected: return 1.0
    if expected == "solid" or detected == "solid": return 0.0
    return 0.5

# Shape type scoring

ACCEPTABLE_TAGS = {
    "circle": {"circle"}, "ellipse": {"ellipse", "circle"},
    "rectangle": {"rect"}, "square": {"rect"},
    "polygon": {"polygon", "path"},
    "cloud": {"path"},
    "3d-cylinder": {"g", "ellipse", "path"},
    "3d-prism": {"g", "rect", "path", "polygon"},
    "3d-cube": {"g", "rect", "polygon"},
    "3d-diamond": {"g", "polygon"}, "3d-hexagon": {"g", "polygon"}, "3d-trapezoid": {"g", "polygon"},
}

def type_score(gt_type, gen_tag, child_tags=None):
    child_tags = child_tags or set()
    ok = ACCEPTABLE_TAGS.get(gt_type, set())
    if gen_tag in ok:
        # For 3D shapes expressed as groups, verify constituent elements
        if gt_type == "3d-cylinder" and gen_tag == "g":
            if "ellipse" not in child_tags: return 0.4  # flat-bottom: no curved caps
        return 1.0
    # Stacked shapes: model wraps the correct element in a <g> for shadow/depth layers
    # (GT also uses <g> for shapes with stacked>1; accept <g> if it has the right child tag)
    if gen_tag == "g" and not gt_type.startswith("3d-") and (child_tags & ok):
        return 1.0
    if gt_type.startswith("3d-"): return 0.4
    if gt_type in ("polygon",) and gen_tag == "rect": return 0.3
    if gt_type in ("rectangle", "square") and gen_tag in ("polygon", "path"): return 0.3
    return 0.0

# Shape extraction

def _collect_texts(root, parent_map):
    """Collect all <text> elements with position, content, and font.

    Applies cumulative translate() offsets from ancestor <g> elements so that
    texts inside <g transform="translate(...)"> groups get correct positions.
    """
    texts = []
    skip_tags = {"defs", "style", "title", "desc", "marker", "filter",
                 "linearGradient", "radialGradient", "pattern", "symbol"}

    def _walk(node, ox=0.0, oy=0.0):
        tag = strip_ns(node.tag)
        if tag in skip_tags:
            return
        if tag == "text":
            x, y = safe_float(node.get("x")), safe_float(node.get("y"))
            txt = collect_text(node).strip()
            if txt:
                texts.append({"x": x + ox, "y": y + oy, "text": txt,
                              "font": get_font(node, parent_map), "used": False})
            return  # don't recurse into tspan children as separate texts
        for child in node:
            child_tag = strip_ns(child.tag)
            if child_tag == "g":
                tx, ty = parse_translate(child.get("transform", ""))
                _walk(child, ox + tx, oy + ty)
            else:
                _walk(child, ox, oy)

    _walk(root)
    return texts

def _detect_overlay_fill(elements, defs, css=None):
    """Check sibling shapes for pattern fill overlay (two-rect technique)."""
    for elem in elements:
        fill = css_get(elem, "fill", css or {})
        if fill.startswith("url(#"):
            ref = fill[5:-1]
            style = defs.get(ref, "")
            if style and style not in ("gradient-linear", "gradient-radial"):
                return style
    return None

def _collect_toplevel_shapes(root, defs, grad_colors, css=None):
    """Collect top-level shape elements (not inside labeled groups)."""
    top = []
    canvas_wh = canvas_size(root)
    defs_map = _build_defs_map(root)
    shape_elems = [(elem, strip_ns(elem.tag)) for elem in root if strip_ns(elem.tag) in SHAPE_TAGS]
    for elem, tag in shape_elems:
        b = bbox(elem, tag)
        if not b or not is_shape(elem, tag, b, css, canvas_wh): continue
        fill_style = detect_fill_style(elem, defs, css)
        # Check for two-rect overlay: if this shape overlaps with a sibling that has a pattern fill
        if fill_style in ("solid", "gradient-linear", "gradient-radial"):
            overlay = _detect_overlay_fill(
                [e for e, _ in shape_elems if e is not elem and _overlaps(b, bbox(e, strip_ns(e.tag)))],
                defs, css)
            if overlay:
                fill_style = overlay
        raw_fill = css_get(elem, "fill", css or {})
        top.append({"elem": elem, "tag": tag, "bounds": b,
                    "fill": resolve_fill_color(raw_fill, grad_colors),
                    "stroke": css_get(elem, "stroke", css or {}), "fill_style": fill_style,
                    "border_style": detect_border_style(elem, css),
                    "child_tags": set()})

    # Also handle top-level <use> elements
    for elem in root:
        if strip_ns(elem.tag) != "use":
            continue
        b = _bbox_use(elem, defs_map)
        if not b or (b[2]-b[0]) < 15 or (b[3]-b[1]) < 15:
            continue
        href = elem.get("href") or elem.get("{http://www.w3.org/1999/xlink}href", "")
        ref = defs_map.get(href.lstrip("#"))
        if ref is None:
            continue
        raw_fill = css_get(ref, "fill", css or {}) or ref.get("fill", "")
        if raw_fill.lower() in ("none", ""):
            continue
        top.append({"elem": elem, "tag": "use", "bounds": b,
                    "fill": resolve_fill_color(raw_fill, grad_colors),
                    "stroke": ref.get("stroke", ""),
                    "fill_style": "solid", "border_style": "solid",
                    "child_tags": set()})
    return top

def _extract_per_shape_labels(elem, ox, oy, css, canvas_wh, grad_colors, defs,
                               parent_map=None, LABEL_PROXIMITY=60):
    """Within a <g>, pair each direct shape child with its nearest text sibling.

    Returns a list of shape dicts only when at least two shapes each have a
    distinct nearby text (i.e., the group contains individually-labeled shapes
    rather than a group-level label).  Returns [] otherwise.
    """
    # Collect direct text-child positions
    local_texts = []
    for child in elem:
        if strip_ns(child.tag) != "text":
            continue
        tx = safe_float(child.get("x", 0)) + ox
        ty = safe_float(child.get("y", 0)) + oy
        t = collect_text(child).strip()
        if t:
            local_texts.append({"x": tx, "y": ty, "text": t})
    if not local_texts:
        return []

    used = set()
    results = []
    for child in elem:
        ctag = strip_ns(child.tag)
        if ctag not in SHAPE_TAGS:
            continue
        cb = bbox(child, ctag, ox, oy)
        if not cb or not is_shape(child, ctag, cb, css, canvas_wh):
            continue
        cx, cy = center(cb)
        best_t, best_d, best_i = None, LABEL_PROXIMITY, -1
        for i, t in enumerate(local_texts):
            if i in used:
                continue
            d = math.sqrt((t["x"] - cx) ** 2 + (t["y"] - cy) ** 2)
            if d < best_d:
                best_d, best_t, best_i = d, t["text"], i
        if best_t is None:
            continue
        used.add(best_i)
        fill   = css_get(child, "fill", css) or child.get("fill", "")
        stroke = css_get(child, "stroke", css) or child.get("stroke", "")
        results.append({
            "label": best_t, "tag": ctag,
            "fill": resolve_fill_color(fill, grad_colors),
            "stroke": stroke, "bounds": cb, "center": center(cb),
            "font": get_font(child, parent_map, css) if parent_map else "",
            "fill_style": detect_fill_style(child, defs, css),
            "border_style": detect_border_style(child, css),
            "child_tags": {ctag},
        })
    # Only return when at least two shapes are individually labeled
    return results if len(results) >= 2 else []


def extract_shapes(root, defs):
    """Extract all shapes from generated SVG."""
    grad_colors = parse_gradient_colors(root)
    css = parse_css(root)
    defs_map = _build_defs_map(root)
    shapes = []
    parent_map = {c: p for p in root.iter() for c in p}
    texts = _collect_texts(root, parent_map)

    # Pattern A: Groups with shape + text children
    canvas_wh = canvas_size(root)
    top_groups = []
    for elem in root.iter():
        if strip_ns(elem.tag) != "g": continue
        ox, oy = parse_translate(elem.get("transform", ""))
        label, bounds, tag, fill, stroke, font = "", None, "", "", "", ""
        fill_style, border_style = "solid", "solid"
        child_tags = set()

        for child in elem:
            ctag = strip_ns(child.tag)
            if ctag == "text":
                t = collect_text(child).strip()
                if t:
                    label = (label + " " + t).strip() if label else t
                    if not font: font = get_font(child, parent_map, css)
            elif ctag == "use":
                # Resolve <use> reference via defs
                cb = _bbox_use(child, defs_map, ox, oy)
                if cb and (cb[2]-cb[0]) >= 15 and (cb[3]-cb[1]) >= 15:
                    child_tags.add("use")
                    bounds = merge_bbox(bounds, cb)
                    if not tag:
                        tag = "use"
                        href = child.get("href") or child.get("{http://www.w3.org/1999/xlink}href", "")
                        ref = defs_map.get(href.lstrip("#"))
                        if ref is not None:
                            fill = css_get(ref, "fill", css) or ref.get("fill", "")
                            stroke = css_get(ref, "stroke", css) or ref.get("stroke", "")
            elif ctag in SHAPE_TAGS:
                child_tags.add(ctag)
                cb = bbox(child, ctag, ox, oy)
                if cb and is_shape(child, ctag, cb, css, canvas_wh):
                    bounds = merge_bbox(bounds, cb)
                    if not tag:
                        tag = ctag
                        fill = css_get(child, "fill", css)
                        stroke = css_get(child, "stroke", css)
                        fill_style = detect_fill_style(child, defs, css)
                        border_style = detect_border_style(child, css)
            elif ctag == "g" and _subtree_has_shape_or_use(child):
                # Nested group (e.g. <g transform><use .../></g>) — union its bounds
                cb = _bbox_group_resolve_uses(child, defs_map, ox, oy)
                if cb and (cb[2]-cb[0]) >= 15 and (cb[3]-cb[1]) >= 15:
                    child_tags.add("g")
                    bounds = merge_bbox(bounds, cb)

        # Two-rect overlay: check siblings for pattern fills that override base fill
        if bounds and fill_style in ("solid", "gradient-linear", "gradient-radial"):
            for child in elem:
                ctag = strip_ns(child.tag)
                if ctag in SHAPE_TAGS:
                    child_fill = css_get(child, "fill", css)
                    if child_fill.startswith("url(#"):
                        ref = child_fill[5:-1]
                        style = defs.get(ref, "")
                        if style and style not in ("gradient-linear", "gradient-radial"):
                            fill_style = style
                            break

        if label and bounds:
            # Check if the group contains individually-labeled shapes (e.g. a
            # grid of circles each adjacent to its own text label).  If so,
            # extract them separately rather than merging the whole group.
            sub = _extract_per_shape_labels(elem, ox, oy, css, canvas_wh,
                                            grad_colors, defs, parent_map)
            if sub:
                for s in sub:
                    bc2 = s["center"]
                    if not any(
                        math.sqrt((bc2[0]-e["center"][0])**2 + (bc2[1]-e["center"][1])**2) < 20
                        for e in shapes
                    ):
                        shapes.append(s)
                        for t in texts:
                            if normalize_label(t["text"]) in normalize_label(s["label"]):
                                t["used"] = True
                continue  # skip the group-level composite

            bc = center(bounds)
            # Deduplicate by position: skip if an existing shape is at nearly
            # the same location (same label at different positions is allowed)
            is_dup = any(
                math.sqrt((bc[0]-s["center"][0])**2 + (bc[1]-s["center"][1])**2) < 20
                for s in shapes
            )
            if not is_dup:
                nl = normalize_label(label)
                shapes.append({"label": label, "tag": tag,
                              "fill": resolve_fill_color(fill, grad_colors),
                              "stroke": stroke, "bounds": bounds, "center": bc,
                              "font": font, "fill_style": fill_style, "border_style": border_style,
                              "child_tags": child_tags})
                for t in texts:
                    if normalize_label(t["text"]) in nl: t["used"] = True
        elif not label and bounds:
            # Group with shapes but no text - needs proximity matching
            biggest, best_area = None, 0
            for child in elem:
                ctag = strip_ns(child.tag)
                if ctag in SHAPE_TAGS:
                    cb = bbox(child, ctag, ox, oy)
                    if cb:
                        area = (cb[2]-cb[0])*(cb[3]-cb[1])
                        if area > best_area:
                            best_area = area
                            biggest = child
            if biggest is not None:
                top_groups.append({"tag": "g", "bounds": bounds, "center": center(bounds),
                    "fill": resolve_fill_color(css_get(biggest, "fill", css), grad_colors),
                    "stroke": css_get(biggest, "stroke", css),
                    "fill_style": detect_fill_style(biggest, defs, css),
                    "border_style": detect_border_style(biggest, css),
                    "child_tags": child_tags})

    # Pattern B: Top-level shapes matched to nearby text
    top = _collect_toplevel_shapes(root, defs, grad_colors, css)
    top.extend(top_groups)

    # Group overlapping elements (3D shapes)
    groups = group_overlapping(top)
    found = set()  # deduplicate Pattern B by label (same label at top-level = same shape)

    for grp in groups:
        gb = grp[0]["bounds"]
        for s in grp[1:]: gb = merge_bbox(gb, s["bounds"])
        gc = center(gb)

        # Find unused text elements inside or near the shape
        nearby = []
        for i, t in enumerate(texts):
            if t["used"]: continue
            d = math.sqrt((t["x"]-gc[0])**2 + (t["y"]-gc[1])**2)
            margin = 30
            inside = (gb[0]-margin <= t["x"] <= gb[2]+margin and
                      gb[1]-margin <= t["y"] <= gb[3]+margin)
            if inside or d < 150:
                nearby.append((d, i))
        nearby.sort()

        if not nearby: continue

        # Take closest text, then grab vertically stacked texts (multi-line labels)
        best_i = nearby[0][1]
        merged_indices = [best_i]
        bx, by = texts[best_i]["x"], texts[best_i]["y"]
        for _, i in nearby[1:]:
            t = texts[i]
            if abs(t["x"] - bx) < 30:
                if any(abs(t["y"] - texts[mi]["y"]) < 40 for mi in merged_indices):
                    merged_indices.append(i)

        merged_indices.sort(key=lambda i: texts[i]["y"])
        label = " ".join(texts[i]["text"] for i in merged_indices)
        font = texts[merged_indices[0]].get("font", "")
        for i in merged_indices:
            texts[i]["used"] = True

        nl = normalize_label(label)
        if nl in found: continue
        found.add(nl)

        biggest = max(grp, key=lambda s: (s["bounds"][2]-s["bounds"][0])*(s["bounds"][3]-s["bounds"][1]))
        tag = biggest["tag"] if len(grp) == 1 else "g"
        merged_child_tags = set().union(*(s.get("child_tags", set()) for s in grp))
        shapes.append({"label": label, "tag": tag, "fill": biggest["fill"], "stroke": biggest["stroke"],
                      "bounds": gb, "center": gc, "font": font,
                      "fill_style": biggest["fill_style"], "border_style": biggest["border_style"],
                      "child_tags": merged_child_tags})

    return shapes

def group_overlapping(elements):
    """Group shapes that substantially overlap (e.g. 3D prism layers, dual-stroke rects).

    Merge rules:
    - Shapes must have actual bounding-box intersection (not just touching).
    - If one shape is mostly inside the other (>80% of smaller) AND they are
      similar in size (size ratio >70%) → same visual element → merge.
    - If BOTH shapes overlap each other by >30% bilaterally → merge.
    - Otherwise keep separate (handles nested shapes AND shapes that just
      barely clip each other's bounding boxes, e.g. a circle vs a nearby rect).
    """
    if not elements: return []
    n = len(elements)
    parent = list(range(n))

    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for i in range(n):
        for j in range(i+1, n):
            a, b = elements[i]["bounds"], elements[j]["bounds"]
            m = 2
            if (a[2]+m < b[0] or b[2]+m < a[0] or a[3]+m < b[1] or b[3]+m < a[1]):
                continue  # no proximity
            ix = min(a[2], b[2]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[1], b[1])
            if ix <= 0 or iy <= 0:
                continue  # touching but no actual area overlap
            inter = ix * iy
            area_a = max(1, (a[2]-a[0]) * (a[3]-a[1]))
            area_b = max(1, (b[2]-b[0]) * (b[3]-b[1]))
            min_area = min(area_a, area_b)
            max_area = max(area_a, area_b)
            overlap_small = inter / min_area
            size_ratio = min_area / max_area
            # nested: smaller shape mostly inside larger but clearly different sizes
            if overlap_small > 0.8 and size_ratio < 0.7:
                continue
            # merge only if significant bilateral overlap
            if overlap_small > 0.3 and inter / max_area > 0.3:
                parent[find(j)] = find(i)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(elements[i])
    return list(groups.values())

# GT extraction and matching

def extract_gt_colors(gt_root):
    grad_colors = parse_gradient_colors(gt_root)
    colors = {}
    for elem in gt_root.iter():
        eid = elem.get("id", "")
        if not eid.startswith("shape_"): continue
        tag = strip_ns(elem.tag)
        if tag == "g":
            best_area, fc, sc = 0, "", ""
            for child in elem:
                cb = bbox(child, strip_ns(child.tag))
                if cb:
                    area = (cb[2]-cb[0])*(cb[3]-cb[1])
                    if area > best_area: best_area, fc, sc = area, child.get("fill", ""), child.get("stroke", "")
            colors[eid] = {"fill": resolve_fill_color(fc, grad_colors), "stroke": sc}
        else:
            colors[eid] = {"fill": resolve_fill_color(elem.get("fill", ""), grad_colors),
                           "stroke": elem.get("stroke", "")}
    return colors

def match_shapes(gen_shapes, gt_shapes):
    matches, used = {}, set()
    for sid, gt in gt_shapes.items():
        gt_label = normalize_label(gt["label"])
        best_i, best_score = None, 0
        for i, gen in enumerate(gen_shapes):
            if i in used: continue
            gen_label = normalize_label(gen["label"])
            if gen_label == gt_label: best_i, best_score = i, 1.0; break
            gw, gtw = set(gen_label.split()), set(gt_label.split())
            if gw and gw.issubset(gtw): s = len(gw)/len(gtw)
            elif gtw and gtw.issubset(gw): s = len(gtw)/len(gw)
            else:
                s = 0
                # Substring/startswith fallback for partial word stems (e.g. "emit" vs "emitter")
                if gen_label in gt_label:
                    s = len(gen_label) / len(gt_label)
                elif gt_label in gen_label:
                    s = len(gt_label) / len(gen_label)
                elif gen_label.startswith(gt_label) or gt_label.startswith(gen_label):
                    shorter = min(len(gen_label), len(gt_label))
                    longer = max(len(gen_label), len(gt_label))
                    s = shorter / longer
            if s > best_score: best_i, best_score = i, s
        if best_i is not None and best_score >= 0.5:
            matches[sid] = gen_shapes[best_i]
            used.add(best_i)
        else:
            matches[sid] = None
    return matches

# Position scoring (anchor-based relative offset ratios)

def find_best_anchor(gt_shapes, matches):
    """Try every matched shape as anchor, return the one giving highest avg position score."""
    candidates = [sid for sid in gt_shapes if matches.get(sid) is not None]
    if not candidates: return None
    if len(candidates) == 1: return candidates[0]

    best_anchor, best_avg = None, -1
    for anchor in candidates:
        total, count = 0.0, 0
        for sid in gt_shapes:
            if matches.get(sid) is None: continue
            total += relative_position_score(sid, gt_shapes, matches, anchor)
            count += 1
        avg = total / count if count > 0 else 0.0
        if avg > best_avg:
            best_avg, best_anchor = avg, anchor
    return best_anchor

def relative_position_score(sid, gt_shapes, matches, anchor_sid):
    """Score position by comparing relative offset ratios from anchor."""
    gen = matches.get(sid)
    if not gen: return 0.0
    if sid == anchor_sid: return 1.0

    gt_anchor_c = center(gt_shapes[anchor_sid]["bounds"])
    gen_anchor = matches[anchor_sid]
    if not gen_anchor: return 0.0
    gen_anchor_c = gen_anchor["center"]

    gt_c = center(gt_shapes[sid]["bounds"])
    gen_c = gen["center"]

    gt_dx = gt_c[0] - gt_anchor_c[0]
    gt_dy = gt_c[1] - gt_anchor_c[1]
    gen_dx = gen_c[0] - gen_anchor_c[0]
    gen_dy = gen_c[1] - gen_anchor_c[1]

    gt_dist = math.sqrt(gt_dx**2 + gt_dy**2)
    gen_dist = math.sqrt(gen_dx**2 + gen_dy**2)

    if gt_dist < 1: return 1.0

    # Direction score
    gt_angle = math.atan2(gt_dy, gt_dx)
    gen_angle = math.atan2(gen_dy, gen_dx)
    angle_diff = abs(gt_angle - gen_angle)
    if angle_diff > math.pi: angle_diff = 2 * math.pi - angle_diff
    dir_score = max(0.0, 1.0 - angle_diff / math.pi)

    # Offset ratio score
    gt_rx = gt_dx / gt_dist
    gt_ry = gt_dy / gt_dist
    if gen_dist < 1:
        ratio_score = 0.0
    else:
        gen_rx = gen_dx / gen_dist
        gen_ry = gen_dy / gen_dist
        vec_diff = math.sqrt((gt_rx - gen_rx)**2 + (gt_ry - gen_ry)**2)
        ratio_score = max(0.0, 1.0 - vec_diff / 2.0)

    return 0.5 * dir_score + 0.5 * ratio_score

def estimate_position_anchor(sid, gt_shapes, matches, anchor_sid, gen_canvas_w, gen_canvas_h):
    """Estimate expected position for GT shape using anchor-based relative positioning."""
    gen_anchor = matches.get(anchor_sid)
    if not gen_anchor: return None

    gt_anchor_c = center(gt_shapes[anchor_sid]["bounds"])
    gen_anchor_c = gen_anchor["center"]

    all_gt_centers = [center(v["bounds"]) for v in gt_shapes.values()]
    gt_min_x = min(c[0] for c in all_gt_centers)
    gt_max_x = max(c[0] for c in all_gt_centers)
    gt_min_y = min(c[1] for c in all_gt_centers)
    gt_max_y = max(c[1] for c in all_gt_centers)
    gt_span_x = gt_max_x - gt_min_x if gt_max_x > gt_min_x else 1
    gt_span_y = gt_max_y - gt_min_y if gt_max_y > gt_min_y else 1

    matched_centers = [m["center"] for m in matches.values() if m is not None]
    if len(matched_centers) < 2:
        scale_x, scale_y = 1.0, 1.0
    else:
        gen_min_x = min(c[0] for c in matched_centers)
        gen_max_x = max(c[0] for c in matched_centers)
        gen_min_y = min(c[1] for c in matched_centers)
        gen_max_y = max(c[1] for c in matched_centers)
        gen_span_x = gen_max_x - gen_min_x if gen_max_x > gen_min_x else 1
        gen_span_y = gen_max_y - gen_min_y if gen_max_y > gen_min_y else 1
        scale_x = gen_span_x / gt_span_x
        scale_y = gen_span_y / gt_span_y

    gt = gt_shapes[sid]
    gt_c = center(gt["bounds"])
    gt_w = gt["bounds"][2] - gt["bounds"][0]
    gt_h = gt["bounds"][3] - gt["bounds"][1]

    dx = (gt_c[0] - gt_anchor_c[0]) * scale_x
    dy = (gt_c[1] - gt_anchor_c[1]) * scale_y

    cx = gen_anchor_c[0] + dx
    cy = gen_anchor_c[1] + dy

    gen = matches.get(sid)
    if gen:
        gw = gen["bounds"][2] - gen["bounds"][0]
        gh = gen["bounds"][3] - gen["bounds"][1]
    else:
        gw = gt_w * scale_x
        gh = gt_h * scale_y

    return [cx - gw/2, cy - gh/2, cx + gw/2, cy + gh/2]

# Aspect ratio scoring

def aspect_ratio_score(gt_bounds, gen_bounds):
    """Score aspect ratio similarity. Returns 1.0 for exact match, falls off as ratios diverge."""
    gt_w = gt_bounds[2] - gt_bounds[0]
    gt_h = gt_bounds[3] - gt_bounds[1]
    gen_w = gen_bounds[2] - gen_bounds[0]
    gen_h = gen_bounds[3] - gen_bounds[1]

    if gt_h < 1 or gen_h < 1 or gt_w < 1 or gen_w < 1:
        return 0.0

    gt_ar = gt_w / gt_h
    gen_ar = gen_w / gen_h
    r = gt_ar / gen_ar if gt_ar > gen_ar else gen_ar / gt_ar

    if r <= 1.2: return 1.0
    if r >= 3.0: return 0.0
    return max(0.0, 1.0 - (r - 1.2) / 1.8)

# Visualization

def draw_annotations(svg_path, gen_shapes, out_path, gt_shapes, matches, shape_scores, gt_to_gen_idx,
                     n_gt=0, n_extra=0, extra_penalty=0.0, composite=0.0):
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    cw, ch = canvas_size(root)

    g = ET.SubElement(root, "g", id="shape-overlay", opacity="0.9")

    anchor_sid = find_best_anchor(gt_shapes, matches)

    # Expected shapes (green dashed)
    for sid, gt in gt_shapes.items():
        if anchor_sid:
            est_bounds = estimate_position_anchor(sid, gt_shapes, matches, anchor_sid, cw, ch)
        else:
            est_bounds = None
        if est_bounds:
            x1, y1, x2, y2 = est_bounds
        else:
            b = gt["bounds"]
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

        r = ET.SubElement(g, "rect", x=str(int(x1)), y=str(int(y1)),
                          width=str(int(x2-x1)), height=str(int(y2-y1)))
        r.set("fill", "rgba(0,255,0,0.1)"); r.set("stroke", "#00FF00"); r.set("stroke-width", "2")
        r.set("stroke-dasharray", "10,5")
        t = ET.SubElement(g, "text", x=str(int(x1)+5), y=str(int(y1)-5))
        t.set("font-family", "monospace"); t.set("font-size", "10"); t.set("fill", "#00FF00"); t.set("font-weight", "bold")
        t.text = f"Expected: {sid}: {gt['label']}"

    gen_idx_to_gt = {v: k for k, v in gt_to_gen_idx.items()}

    # Detected shapes (red dashed)
    for i, s in enumerate(gen_shapes):
        b = s["bounds"]
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

        r = ET.SubElement(g, "rect", x=str(x1), y=str(y1), width=str(x2-x1), height=str(y2-y1))
        r.set("fill", "none"); r.set("stroke", "#FF0000"); r.set("stroke-width", "2")
        r.set("stroke-dasharray", "5,5")

        sc = shape_scores.get(i, {})
        matched_gt = gen_idx_to_gt.get(i)

        if matched_gt:
            lines = [f"{matched_gt}: {s['label'][:20]}"]
        else:
            lines = [f"[unmatched] {s['label'][:20]}"]
        line_scores = [None]
        lines.append(f"Type: {s['tag']} ({sc.get('type',0):.2f})")
        line_scores.append(sc.get('type') if sc else None)
        lines.append(f"Fill: {s.get('fill','?')[:10]} ({sc.get('fill_color',0):.2f})")
        line_scores.append(sc.get('fill_color') if sc else None)
        lines.append(f"Style: {s.get('fill_style','?')} ({sc.get('fill_style',0):.2f})")
        line_scores.append(sc.get('fill_style') if sc else None)
        lines.append(f"Pos: ({int(s['center'][0])},{int(s['center'][1])}) ({sc.get('position',0):.2f})")
        line_scores.append(sc.get('position') if sc else None)
        lines.append(f"AR: {sc.get('aspect_ratio',0):.2f}")
        line_scores.append(sc.get('aspect_ratio') if sc else None)
        if sc:
            lines.append(f"Composite: {sc.get('composite',0):.3f}")
            line_scores.append(sc.get('composite'))

        ly = max(y1 - len(lines)*14 - 5, 5)
        bw = max(len(l) for l in lines) * 7 + 10
        bg = ET.SubElement(g, "rect", x=str(x1-2), y=str(ly-12), width=str(bw), height=str(len(lines)*14+8))
        bg.set("fill", "white"); bg.set("stroke", "#FF0000"); bg.set("stroke-width", "1"); bg.set("opacity", "0.95")

        for j, (line, lscore) in enumerate(zip(lines, line_scores)):
            t = ET.SubElement(g, "text", x=str(x1+2), y=str(ly + j*14))
            t.set("font-family", "monospace"); t.set("font-size", "11")
            t.set("fill", "#060" if lscore is not None and lscore >= 0.8 else
                          "#960" if lscore is not None and lscore >= 0.5 else
                          "#C00" if lscore is not None else "#000")
            t.text = line

    # Penalty summary
    if n_extra > 0:
        box_w, box_h = 350, 40
        bx, by = cw - box_w - 10, ch - box_h - 10
        pbg = ET.SubElement(g, "rect", x=str(bx), y=str(by), width=str(box_w), height=str(box_h))
        pbg.set("fill", "white"); pbg.set("stroke", "#C00"); pbg.set("stroke-width", "2"); pbg.set("rx", "4"); pbg.set("opacity", "0.95")
        pt = ET.SubElement(g, "text", x=str(bx+8), y=str(by+16))
        pt.set("font-family", "monospace"); pt.set("font-size", "10"); pt.set("font-weight", "bold"); pt.set("fill", "#C00")
        pt.text = f"GT:{n_gt} Det:{len(gen_shapes)} Extra:{n_extra}"
        pt2 = ET.SubElement(g, "text", x=str(bx+8), y=str(by+32))
        pt2.set("font-family", "monospace"); pt2.set("font-size", "10"); pt2.set("font-weight", "bold"); pt2.set("fill", "#C00")
        pt2.text = f"Penalty:-{extra_penalty:.3f} Composite:{composite:.3f}"

    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)

# Scoring

_SHAPE_METRICS = ["label", "type", "fill_color", "fill_style", "stroke_color",
                  "border_style", "position", "font", "aspect_ratio"]
WEIGHTS = {k: 1.0 / len(_SHAPE_METRICS) for k in _SHAPE_METRICS}

def score_single_shape(sid, gt, gen, gt_col, gt_shapes, matches, anchor_sid):
    """Score a single matched shape across all metrics. Returns dict of metric scores."""
    gt_b = gt["bounds"]
    gen_b = gen["bounds"]

    gen_nl, gt_nl = normalize_label(gen["label"]), normalize_label(gt["label"])
    if gen_nl == gt_nl: label_sc = 1.0
    elif gen_nl in gt_nl or gt_nl in gen_nl:
        label_sc = min(len(gen_nl), len(gt_nl)) / max(len(gen_nl), len(gt_nl))
    else:
        gw, gtw = set(gen_nl.split()), set(gt_nl.split())
        common = gw & gtw
        label_sc = len(common) / max(len(gw), len(gtw)) if common else 0.0
    type_sc = type_score(gt["type"], gen["tag"], gen.get("child_tags"))
    # Prefer JSON metadata fields (fillColor/strokeColor) over SVG-extracted colours.
    # GT SVG extraction can return blended/pattern-overlay colours for 3D shapes and
    # pattern-filled shapes (url(#...) refs), causing false-low scores.
    gt_fill_c   = gt.get("fillColor")   or gt_col.get("fill",   "")
    gt_stroke_c = gt.get("strokeColor") or gt_col.get("stroke", "")
    fill_col_sc   = color_score(gt_fill_c,   gen["fill"])   if gt_fill_c   else 0.5
    stroke_col_sc = color_score(gt_stroke_c, gen["stroke"]) if gt_stroke_c else 0.5
    fill_sty_sc = fill_style_score(gt.get("fillStyle", "solid"), gen.get("fill_style", "solid"))
    border_sty_sc = border_style_score(gt.get("borderStyle", "solid"), gen.get("border_style", "solid"))
    pos_sc = relative_position_score(sid, gt_shapes, matches, anchor_sid) if anchor_sid else 0.0
    font_sc = font_score(gt.get("font", ""), gen.get("font", "")) if gt.get("font") else 0.5
    ar_sc = aspect_ratio_score(gt_b, gen_b)

    return {
        "label": label_sc, "type": type_sc, "fill_color": fill_col_sc,
        "fill_style": fill_sty_sc, "stroke_color": stroke_col_sc,
        "border_style": border_sty_sc, "position": pos_sc, "font": font_sc,
        "aspect_ratio": ar_sc,
    }

# Public API

def evaluate_shapes(sample_id, objects_dir, gt_svg_dir, model_output_dir, ann_dir=None):
    """Evaluate shapes for one sample (directory-based). Returns (scores_dict, report_string)."""
    objects_dir, gt_svg_dir, model_output_dir = Path(objects_dir), Path(gt_svg_dir), Path(model_output_dir)
    return evaluate_shapes_file(
        gen_svg=model_output_dir / f"{sample_id}.svg",
        gt_svg=gt_svg_dir / f"{sample_id}.svg",
        meta_json=objects_dir / f"{sample_id}.json",
        ann_path=Path(ann_dir) / f"{sample_id}_shapes.svg" if ann_dir else None,
        label=sample_id,
    )

def evaluate_shapes_file(gen_svg, gt_svg, meta_json, ann_path=None, label=None, ann_base_svg=None):
    """Evaluate shapes for one SVG file. Returns (scores_dict, report_string).

    Args:
        gen_svg:   Path to the generated (model output) SVG.
        gt_svg:    Path to the ground-truth SVG.
        meta_json: Path to the metadata JSON (entities dict with bounds, labels, types).
        ann_path:  Optional path to write an annotated SVG overlay.
        label:     Optional label for the report header (defaults to gen_svg stem).
    """
    gen_svg, gt_svg, meta_json = Path(gen_svg), Path(gt_svg), Path(meta_json)
    sample_id = label or gen_svg.stem

    meta = json.loads(meta_json.read_text("utf-8"))
    gt_shapes = {k: v for k, v in meta["entities"].items() if v.get("type") != "arrow"}

    gt_root = ET.parse(str(gt_svg)).getroot()
    gt_colors = extract_gt_colors(gt_root)

    gen_path = gen_svg
    gen_root = ET.parse(str(gen_path)).getroot()
    defs = parse_defs(gen_root)
    gen_shapes = extract_shapes(gen_root, defs)

    matches = match_shapes(gen_shapes, gt_shapes)

    gt_to_gen_idx = {}
    for sid, gen in matches.items():
        if gen is not None:
            idx = next((i for i, s in enumerate(gen_shapes) if s is gen), None)
            if idx is not None:
                gt_to_gen_idx[sid] = idx

    anchor_sid = find_best_anchor(gt_shapes, matches)

    cat_scores = {k: [] for k in WEIGHTS}
    shape_scores = {}

    gt_labels = [f"{k}: {normalize_label(gt_shapes[k]['label'])}" for k in sorted(gt_shapes)]
    gen_labels = [normalize_label(s["label"]) for s in gen_shapes]

    report = [f"{'='*50}", f"SHAPE EVAL: {sample_id}",
              f"Expected: {len(gt_shapes)} ({', '.join(gt_labels[:5])}{'...' if len(gt_labels)>5 else ''})",
              f"Detected: {len(gen_shapes)} ({', '.join(gen_labels[:5])}{'...' if len(gen_labels)>5 else ''})",
              f"Anchor: {anchor_sid} ({gt_shapes[anchor_sid]['label'] if anchor_sid else 'none'})", ""]

    for sid in sorted(gt_shapes):
        gt = gt_shapes[sid]
        gen = matches[sid]
        gt_col = gt_colors.get(sid, {})
        gen_idx = gt_to_gen_idx.get(sid)

        report.append(f"--- {sid}: {gt['label']} ---")
        gt_b = gt["bounds"]
        gt_w, gt_h = gt_b[2] - gt_b[0], gt_b[3] - gt_b[1]
        gt_ar = gt_w / gt_h if gt_h > 0 else 1.0
        report.append(f"  GT type={gt['type']}  fillStyle={gt.get('fillStyle','solid')}  borderStyle={gt.get('borderStyle','solid')}  font={gt.get('font','')}")
        report.append(f"  GT fill={gt_col.get('fill','')}  stroke={gt_col.get('stroke','')}  bounds={gt_b}  AR={gt_ar:.2f}")

        if not gen:
            report.append("  MATCH: NOT FOUND -- all scores 0.0")
            for k in cat_scores: cat_scores[k].append(0.0)
            continue

        gen_b = gen["bounds"]
        gen_w, gen_h = gen_b[2] - gen_b[0], gen_b[3] - gen_b[1]
        gen_ar = gen_w / gen_h if gen_h > 0 else 1.0
        report.append(f"  GEN [{gen_idx}] label={gen['label']}  tag={gen['tag']}  fill={gen['fill']}  stroke={gen['stroke']}")
        report.append(f"  GEN fillStyle={gen.get('fill_style','solid')}  borderStyle={gen.get('border_style','solid')}  font={gen.get('font','')}")
        report.append(f"  GEN center=({gen['center'][0]:.0f},{gen['center'][1]:.0f})  bounds=[{gen_b[0]:.0f},{gen_b[1]:.0f},{gen_b[2]:.0f},{gen_b[3]:.0f}]  AR={gen_ar:.2f}")

        scores = score_single_shape(sid, gt, gen, gt_col, gt_shapes, matches, anchor_sid)
        for k, v in scores.items():
            cat_scores[k].append(v)

        comp = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
        if gen_idx is not None:
            shape_scores[gen_idx] = {"composite": comp, **scores}

        w_pct = f"{WEIGHTS['label']:.0%}"
        report.append(f"  SCORES:")
        report.append(f"    label:        {scores['label']:.3f}  (weight {w_pct})  {'exact' if scores['label'] == 1.0 else 'partial'} match")
        report.append(f"    type:         {scores['type']:.3f}  (weight {w_pct})  GT={gt['type']} GEN={gen['tag']}")
        report.append(f"    fill_color:   {scores['fill_color']:.3f}  (weight {w_pct})  GT={gt_col.get('fill','')} GEN={gen['fill']}")
        report.append(f"    fill_style:   {scores['fill_style']:.3f}  (weight {w_pct})  GT={gt.get('fillStyle','solid')} GEN={gen.get('fill_style','solid')}")
        report.append(f"    stroke_color: {scores['stroke_color']:.3f}  (weight {w_pct})  GT={gt_col.get('stroke','')} GEN={gen['stroke']}")
        report.append(f"    border_style: {scores['border_style']:.3f}  (weight {w_pct})  GT={gt.get('borderStyle','solid')} GEN={gen.get('border_style','solid')}")
        report.append(f"    position:     {scores['position']:.3f}  (weight {w_pct})  relative offset from anchor {anchor_sid}")
        report.append(f"    font:         {scores['font']:.3f}  (weight {w_pct})  GT={gt.get('font','')} GEN={gen.get('font','')}")
        report.append(f"    aspect_ratio: {scores['aspect_ratio']:.3f}  (weight {w_pct})  GT={gt_ar:.2f} GEN={gen_ar:.2f}")
        report.append(f"  COMPOSITE:      {comp:.3f}")

    # Extra shapes penalty
    matched_labels = set()
    for sid2, gen in matches.items():
        if gen: matched_labels.add(normalize_label(gen["label"]))
    extra_shapes = [s for s in gen_shapes if normalize_label(s["label"]) not in matched_labels]
    n_gt = len(gt_shapes)
    n_extra = len(extra_shapes)
    extra_penalty = n_extra / (n_gt + n_extra) if (n_gt + n_extra) > 0 else 0.0

    if extra_shapes:
        report.append("")
        report.append(f"--- EXTRA SHAPES (not in GT): {n_extra} ---")
        for s in extra_shapes:
            report.append(f"  '{s['label']}'  tag={s['tag']}  center=({s['center'][0]:.0f},{s['center'][1]:.0f})")
        report.append(f"  Penalty: {extra_penalty:.3f} (reduces composite)")

    avg = {k: sum(v)/len(v) if v else 0.0 for k, v in cat_scores.items()}
    raw_composite = sum(avg[k] * WEIGHTS[k] for k in WEIGHTS)
    composite = raw_composite * (1.0 - extra_penalty)

    report.append("")
    report.append(f"{'='*50}")
    report.append("CATEGORY AVERAGES:")
    for k in WEIGHTS:
        v = avg[k]
        report.append(f"  {k:<15} {v:.3f}  (weight {WEIGHTS[k]:.0%})")
    report.append(f"{'='*50}")
    if n_extra > 0:
        report.append(f"RAW COMPOSITE:   {raw_composite:.3f}")
        report.append(f"EXTRA PENALTY:   -{extra_penalty:.3f}  ({n_extra} extra shapes / {n_gt + n_extra} total)")
    report.append(f"COMPOSITE: {composite:.3f}")

    if ann_path:
        ann_path = Path(ann_path)
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        base_svg = Path(ann_base_svg) if ann_base_svg else gen_path
        draw_annotations(base_svg, gen_shapes, ann_path,
                        gt_shapes, matches, shape_scores, gt_to_gen_idx,
                        n_gt, n_extra, extra_penalty, composite)

    return {**avg, "extra_penalty": extra_penalty, "composite": composite}, "\n".join(report)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate shapes for one SVG pair.")
    ap.add_argument("gen_svg",    help="Generated (model output) SVG")
    ap.add_argument("gt_svg",     help="Ground-truth SVG")
    ap.add_argument("meta_json",  help="Ground-truth metadata JSON")
    ap.add_argument("--annotate", metavar="OUT_SVG", default=None,
                    help="Write annotated overlay SVG to this path")
    a = ap.parse_args()
    scores, report = evaluate_shapes_file(a.gen_svg, a.gt_svg, a.meta_json, ann_path=a.annotate)
    print(report)
    print(f"\nComposite: {scores['composite']:.3f}")
