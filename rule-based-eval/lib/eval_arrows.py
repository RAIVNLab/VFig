#!/usr/bin/env python3
"""Arrow evaluation module - scores generated SVG arrows against ground truth.

Can be run standalone to evaluate a single SVG pair:
    python lib/eval_arrows.py generated.svg gt.svg metadata.json
    python lib/eval_arrows.py generated.svg gt.svg metadata.json --annotate out.svg
"""

import sys, json, math, re
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import strip_ns, safe_float, find_nums, parse_translate, normalize_label, hex_to_rgb, color_score, canvas_size
from eval_shapes import extract_shapes, parse_defs

# CSS parsing and transform accumulation

def parse_css(root):
    styles = {}
    for elem in root.iter():
        if strip_ns(elem.tag) == "style":
            for m in re.finditer(r'([.#]?[\w-]+)\s*\{([^}]*)\}', elem.text or ""):
                sel = m.group(1).lstrip(".")
                props = {pm.group(1).strip(): pm.group(2).strip()
                        for pm in re.finditer(r'([\w-]+)\s*:\s*([^;]+)', m.group(2))}
                styles[sel] = props
    return styles

def accumulate_transform(elem, parent_map):
    tx, ty = 0.0, 0.0
    cur = elem
    while cur in parent_map:
        cur = parent_map[cur]
        dx, dy = parse_translate(cur.get("transform", ""))
        tx, ty = tx + dx, ty + dy
    dx, dy = parse_translate(elem.get("transform", ""))
    return tx + dx, ty + dy

def get_inherited(elem, attr, parent_map, css):
    val = elem.get(attr, "")
    if val: return val
    for cls in elem.get("class", "").split():
        if cls in css and attr in css[cls]: return css[cls][attr]
    cur = elem
    while cur in parent_map:
        cur = parent_map[cur]
        if strip_ns(cur.tag) == "g":
            val = cur.get(attr, "")
            if val: return val
            for cls in cur.get("class", "").split():
                if cls in css and attr in css[cls]: return css[cls][attr]
    return ""

# Path analysis

def path_start(d):
    m = re.match(r"[Mm]\s*([-\d.]+)[,\s]+([-\d.]+)", d.strip())
    return (float(m.group(1)), float(m.group(2))) if m else None

def path_end(d):
    tokens = re.findall(r"[MmLlHhVvQqCcSsTtAaZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    x, y, sx, sy, i, cmd = 0, 0, 0, 0, 0, "M"
    while i < len(tokens):
        if tokens[i].isalpha(): cmd, i = tokens[i], i+1; continue
        rel, uc = cmd.islower(), cmd.upper()
        try:
            if uc == "M" and i+1 < len(tokens):
                nx, ny = float(tokens[i]), float(tokens[i+1])
                x, y = (x+nx, y+ny) if rel else (nx, ny); sx, sy = x, y; i += 2
            elif uc == "L" and i+1 < len(tokens):
                nx, ny = float(tokens[i]), float(tokens[i+1])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 2
            elif uc == "H" and i < len(tokens):
                x = x + float(tokens[i]) if rel else float(tokens[i]); i += 1
            elif uc == "V" and i < len(tokens):
                y = y + float(tokens[i]) if rel else float(tokens[i]); i += 1
            elif uc == "Q" and i+3 < len(tokens):
                nx, ny = float(tokens[i+2]), float(tokens[i+3])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 4
            elif uc == "C" and i+5 < len(tokens):
                nx, ny = float(tokens[i+4]), float(tokens[i+5])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 6
            elif uc == "S" and i+3 < len(tokens):
                nx, ny = float(tokens[i+2]), float(tokens[i+3])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 4
            elif uc == "T" and i+1 < len(tokens):
                nx, ny = float(tokens[i]), float(tokens[i+1])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 2
            elif uc == "A" and i+6 < len(tokens):
                nx, ny = float(tokens[i+5]), float(tokens[i+6])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 7
            elif uc == "Z": x, y = sx, sy; i += 1
            else: i += 1
        except (ValueError, IndexError): i += 1
    return (x, y)

def path_length(d):
    """Approximate path length by summing segment distances."""
    tokens = re.findall(r"[MmLlHhVvQqCcSsTtAaZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    x, y, sx, sy, i, cmd, length = 0, 0, 0, 0, 0, "M", 0.0
    while i < len(tokens):
        if tokens[i].isalpha(): cmd, i = tokens[i], i+1; continue
        rel, uc = cmd.islower(), cmd.upper()
        px, py = x, y
        try:
            if uc == "M" and i+1 < len(tokens):
                nx, ny = float(tokens[i]), float(tokens[i+1])
                x, y = (x+nx, y+ny) if rel else (nx, ny); sx, sy = x, y; i += 2
            elif uc == "L" and i+1 < len(tokens):
                nx, ny = float(tokens[i]), float(tokens[i+1])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 2
            elif uc == "H" and i < len(tokens):
                x = x + float(tokens[i]) if rel else float(tokens[i]); i += 1
            elif uc == "V" and i < len(tokens):
                y = y + float(tokens[i]) if rel else float(tokens[i]); i += 1
            elif uc == "Q" and i+3 < len(tokens):
                nx, ny = float(tokens[i+2]), float(tokens[i+3])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 4
            elif uc == "C" and i+5 < len(tokens):
                nx, ny = float(tokens[i+4]), float(tokens[i+5])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 6
            elif uc == "S" and i+3 < len(tokens):
                nx, ny = float(tokens[i+2]), float(tokens[i+3])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 4
            elif uc == "T" and i+1 < len(tokens):
                nx, ny = float(tokens[i]), float(tokens[i+1])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 2
            elif uc == "A" and i+6 < len(tokens):
                nx, ny = float(tokens[i+5]), float(tokens[i+6])
                x, y = (x+nx, y+ny) if rel else (nx, ny); i += 7
            elif uc == "Z": x, y = sx, sy; i += 1
            else: i += 1
        except (ValueError, IndexError): i += 1
        if uc != "M":
            length += math.sqrt((x - px)**2 + (y - py)**2)
    return length

def detect_curvature(d):
    tokens = re.findall(r"[MmLlHhVvQqCcSsTtAaZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    curves = set()
    for t in tokens:
        if t.upper() in "QTS": curves.add("quadratic")
        elif t.upper() == "C": curves.add("cubic")
        elif t.upper() == "A": curves.add("arc")

    if not curves: return {"curved": False, "type": "straight", "amount": 0.0}

    start, end = path_start(d), path_end(d)
    if not start or not end: return {"curved": True, "type": list(curves)[0], "amount": 0.2}
    return {"curved": True, "type": list(curves)[0] if len(curves) == 1 else "mixed",
            "amount": min(0.5, 0.2)}

def detect_dash(elem, parent_map, css):
    da = get_inherited(elem, "stroke-dasharray", parent_map, css)
    if not da or da.lower() == "none": return "none"
    nums = find_nums(da)
    if not nums: return "none"
    if len(nums) >= 4: return "dash-dot"
    if nums[0] <= 3: return "dotted"
    return "dashed"

# Arrowhead extraction

def extract_markers(root):
    markers = {}
    for elem in root.iter():
        if strip_ns(elem.tag) == "marker":
            mid = elem.get("id", "")
            mw = safe_float(elem.get("markerWidth"), 10)
            mh = safe_float(elem.get("markerHeight"), 10)
            units = elem.get("markerUnits", "strokeWidth")

            vw, vh = mh, mw
            for child in elem:
                ctag = strip_ns(child.tag)
                if ctag == "polygon":
                    pts = find_nums(child.get("points", ""))
                    if len(pts) >= 6:
                        xs = [pts[i] for i in range(0, len(pts), 2)]
                        ys = [pts[i] for i in range(1, len(pts), 2)]
                        vh, vw = max(xs)-min(xs), max(ys)-min(ys)
                elif ctag == "path":
                    pts = find_nums(child.get("d", ""))
                    if len(pts) >= 4:
                        xs, ys = pts[0::2], pts[1::2]
                        vh, vw = max(xs)-min(xs), max(ys)-min(ys)

            markers[mid] = {"mw": mw, "mh": mh, "units": units, "vw": vw, "vh": vh}
    return markers

def _polygon_tip(pts_flat):
    """Return the tip vertex of an arrowhead polygon.

    For a standard 3-point arrowhead the tip is the vertex that is farthest
    from the midpoint of the other two (i.e. the lone 'sharp' point).
    Works for polygons with 3 or more vertices.
    """
    if len(pts_flat) < 6:
        return None
    pts = [(pts_flat[i], pts_flat[i + 1]) for i in range(0, len(pts_flat) - 1, 2)]
    best_d, tip = -1.0, None
    for i in range(len(pts)):
        others = [pts[j] for j in range(len(pts)) if j != i]
        mx = sum(o[0] for o in others) / len(others)
        my = sum(o[1] for o in others) / len(others)
        d = math.sqrt((pts[i][0] - mx) ** 2 + (pts[i][1] - my) ** 2)
        if d > best_d:
            best_d, tip = d, pts[i]
    return tip


def extract_gt_arrowheads(gt_root, gt_shape_bboxes=None):
    """Return per-arrow dicts with size info and visibility.

    Args:
        gt_root:          ElementTree root of the GT SVG.
        gt_shape_bboxes:  List of [x1,y1,x2,y2] shape bounding boxes from the
                          JSON metadata.  When supplied, each entry gains a
                          ``head_visible`` bool that is False when the arrowhead
                          tip falls inside a shape (i.e. the arrowhead is drawn
                          but hidden behind the shape layer).

    Returns list ordered by document appearance (same order as sorted(gt_arrows)
    for programmatically-generated datasets).
    """
    sizes = []
    for elem in gt_root:
        if strip_ns(elem.tag) != "g": continue
        has_path = any(strip_ns(c.tag) in ("path", "line") and
                      (c.get("fill") or "").lower() in ("", "none") for c in elem)
        poly = None
        sw = 2.0
        for c in elem:
            ctag = strip_ns(c.tag)
            if ctag == "polygon": poly = c
            if ctag in ("path", "line"):
                sw = safe_float(c.get("stroke-width"), 2)
        if has_path and poly is not None:
            pts_flat = find_nums(poly.get("points", ""))
            if len(pts_flat) >= 6:
                xs = [pts_flat[i] for i in range(0, len(pts_flat), 2)]
                ys = [pts_flat[i] for i in range(1, len(pts_flat), 2)]
                entry = {"width": max(ys) - min(ys), "height": max(xs) - min(xs), "sw": sw,
                         "head_visible": True}
                if gt_shape_bboxes is not None:
                    tip = _polygon_tip(pts_flat)
                    if tip is not None:
                        tx, ty = tip
                        # A 2 px margin absorbs sub-pixel rendering differences.
                        # Only truly hidden when the tip is clearly inside the shape
                        # interior (≥10px from edge). Tips touching the boundary
                        # are still visible in the rendered SVG.
                        inset = 10
                        hidden = any(
                            b[0] + inset <= tx <= b[2] - inset
                            and b[1] + inset <= ty <= b[3] - inset
                            for b in gt_shape_bboxes
                        )
                        entry["head_visible"] = not hidden
                sizes.append(entry)
    return sizes

# Arrow extraction

def _sibling_polygon_head(elem, parent_map):
    """Check if element has a <polygon> sibling (inline arrowhead)."""
    parent = parent_map.get(elem)
    if parent is None or strip_ns(parent.tag) != "g":
        return False, 0
    for sib in parent:
        if strip_ns(sib.tag) == "polygon":
            pts = find_nums(sib.get("points", ""))
            if len(pts) >= 6:
                ys = [pts[i] for i in range(1, len(pts), 2)]
                return True, max(ys) - min(ys)
    return False, 0

def _is_closed_path(d):
    """Check if path d-string ends with Z/z (closed path = shape outline, not arrow)."""
    stripped = (d or "").rstrip()
    return stripped.endswith("Z") or stripped.endswith("z")

def extract_arrows(root):
    arrows = []
    markers = extract_markers(root)
    css = parse_css(root)
    parent_map = {c: p for p in root.iter() for c in p}

    for elem in root.iter():
        tag = strip_ns(elem.tag)

        if tag == "path":
            d = elem.get("d", "")

            # Skip closed paths (shape outlines, not arrows)
            if _is_closed_path(d): continue

            start, end = path_start(d), path_end(d)
            if not start or not end: continue
            if abs(start[0]-end[0]) < 1 and abs(start[1]-end[1]) < 1: continue

            fill = get_inherited(elem, "fill", parent_map, css).lower()
            if fill and fill != "none": continue

            stroke = get_inherited(elem, "stroke", parent_map, css)
            if not stroke: continue

            ox, oy = accumulate_transform(elem, parent_map)
            start = (start[0]+ox, start[1]+oy)
            end = (end[0]+ox, end[1]+oy)

            # Skip very short segments (< 20px straight-line distance)
            seg_len = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            if seg_len < 20: continue

            sw = safe_float(get_inherited(elem, "stroke-width", parent_map, css), 1)
            marker_end = elem.get("marker-end", "") or get_inherited(elem, "marker-end", parent_map, css)
            has_head = "url(#" in marker_end
            marker_info = None
            if has_head:
                m = re.search(r"url\(#([^)]+)\)", marker_end)
                if m: marker_info = markers.get(m.group(1))

            if not has_head:
                sib_head, sib_w = _sibling_polygon_head(elem, parent_map)
                if sib_head:
                    has_head = True
                    marker_info = {"vw": sib_w, "vh": 0, "mw": 0, "mh": 0, "units": "userSpaceOnUse"}

            arrows.append({
                "start": start, "end": end, "color": stroke, "sw": sw,
                "has_head": has_head, "marker": marker_info,
                "curve": detect_curvature(d), "dash": detect_dash(elem, parent_map, css),
                "_d": d, "_length": path_length(d),
            })

        elif tag == "line":
            x1 = safe_float(elem.get("x1")); y1 = safe_float(elem.get("y1"))
            x2 = safe_float(elem.get("x2")); y2 = safe_float(elem.get("y2"))

            stroke = get_inherited(elem, "stroke", parent_map, css)
            if not stroke: continue

            ox, oy = accumulate_transform(elem, parent_map)
            start, end = (x1+ox, y1+oy), (x2+ox, y2+oy)

            # Skip very short segments (< 20px straight-line distance)
            seg_len = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            if seg_len < 20: continue

            sw = safe_float(get_inherited(elem, "stroke-width", parent_map, css), 1)
            marker_end = elem.get("marker-end", "") or get_inherited(elem, "marker-end", parent_map, css)
            has_head = "url(#" in marker_end
            marker_info = None
            if has_head:
                m = re.search(r"url\(#([^)]+)\)", marker_end)
                if m: marker_info = markers.get(m.group(1))

            if not has_head:
                sib_head, sib_w = _sibling_polygon_head(elem, parent_map)
                if sib_head:
                    has_head = True
                    marker_info = {"vw": sib_w, "vh": 0, "mw": 0, "mh": 0, "units": "userSpaceOnUse"}

            arrows.append({
                "start": start, "end": end, "color": stroke, "sw": sw,
                "has_head": has_head, "marker": marker_info,
                "curve": {"curved": False, "type": "straight", "amount": 0},
                "dash": detect_dash(elem, parent_map, css),
                "_d": None, "_length": seg_len,
            })

    return arrows

def filter_arrows_inside_shapes(arrows, shape_bboxes, threshold=0.8):
    """Remove arrows that lie mostly inside a single shape bbox, and short
    paths that are less than 15% of the nearest shape's diagonal."""
    filtered = []
    for arr in arrows:
        sx, sy = arr["start"]
        ex, ey = arr["end"]
        inside_any = False

        for b in shape_bboxes:
            n_inside = 0
            n_samples = 10
            for t in range(n_samples + 1):
                frac = t / n_samples
                px = sx + frac * (ex - sx)
                py = sy + frac * (ey - sy)
                if b[0] <= px <= b[2] and b[1] <= py <= b[3]:
                    n_inside += 1
            if n_inside / (n_samples + 1) >= threshold:
                inside_any = True
                break

        if inside_any:
            continue

        # Short path filter: reject paths that are inside a shape and clearly
        # too short to be a meaningful connector (< 15% of the shape diagonal).
        # Only applies when the arrow midpoint is strictly inside the shape —
        # arrows near but outside a large shape are legitimate connectors.
        plen = arr.get("_length", 0)
        if plen > 0:
            arr_cx, arr_cy = (sx + ex) / 2, (sy + ey) / 2
            for b in shape_bboxes:
                diag = math.sqrt((b[2]-b[0])**2 + (b[3]-b[1])**2)
                inside = (b[0] <= arr_cx <= b[2] and b[1] <= arr_cy <= b[3])
                if inside and diag > 0 and plen < 0.15 * diag:
                    inside_any = True
                    break

        if not inside_any:
            filtered.append(arr)
    return filtered

# Matching and scoring

def dist_to_bbox(x, y, b, margin=10):
    xmin, ymin, xmax, ymax = b[0]-margin, b[1]-margin, b[2]+margin, b[3]+margin
    if xmin <= x <= xmax and ymin <= y <= ymax: return 0
    dx = max(xmin-x, 0, x-xmax)
    dy = max(ymin-y, 0, y-ymax)
    return math.sqrt(dx**2 + dy**2)

def find_shape_at(x, y, shapes, max_dist=100):
    best, best_d = None, max_dist
    for label, info in shapes.items():
        d = dist_to_bbox(x, y, info["bounds"])
        if d < best_d: best, best_d = label, d
    return best, best_d

def point_inside(x, y, b):
    margin = 5.0
    return b[0]+margin < x < b[2]-margin and b[1]+margin < y < b[3]-margin

def overlap_score(arrow, shapes, from_label, to_label):
    """Penalize only endpoints that land inside a shape they shouldn't be in.
    Endpoints inside the correct source/dest shape are expected and not penalised."""
    sx, sy = arrow["start"]
    ex, ey = arrow["end"]
    bad = 0
    for label, info in shapes.items():
        if point_inside(sx, sy, info["bounds"]) and label != from_label: bad += 1
        if point_inside(ex, ey, info["bounds"]) and label != to_label: bad += 1
    if bad == 0: return 1.0
    if bad == 1: return 0.4
    return 0.0

def arrowhead_size_score(arrow, gt_width, gt_sw=2.0):
    """Score arrowhead size.

    Bug-fix: previous code multiplied the marker's viewport dimension (in
    strokeWidth-units) by the generated stroke-width and compared against the
    GT polygon's pixel width.  This always produced a huge ratio because the
    two quantities are in incompatible units.

    Fix: normalise both sides to stroke-width-relative units before comparing.
      - Marker (markerUnits=strokeWidth): vw is already in sw-units.
      - Marker (markerUnits=userSpaceOnUse): divide by stroke-width to convert.
      - GT inline polygon: gt_width / gt_sw gives the sw-relative size.
    """
    if not arrow["has_head"] or not arrow["marker"]: return 0.0
    m = arrow["marker"]
    sw = arrow["sw"] if arrow["sw"] > 0 else 1.0
    # Convert model arrowhead to stroke-width units
    if m["units"] == "strokeWidth":
        model_vw = m["vw"]          # already sw-relative
    else:
        model_vw = m["vw"] / sw     # userSpaceOnUse → sw-relative
    # Convert GT polygon to stroke-width units
    gt_ratio = (gt_width / gt_sw) if gt_sw > 0 else gt_width
    if gt_ratio == 0: gt_ratio = 5.0
    if model_vw == 0: return 0.0
    r = model_vw / gt_ratio if model_vw >= gt_ratio else gt_ratio / model_vw
    # Tolerance tiers (r is always >= 1.0):
    #   r <= 1.3  (+/-30%): within normal SVG renderer rounding and refX/refY
    #             placement variation — considered correct.
    #   r <= 1.8  (+/-80%): noticeably larger/smaller but the same visual class
    #             of arrowhead — partial credit.
    #   r <= 2.5  (+/-150%): clearly disproportionate; minimal credit.
    #   r >  2.5: arrowhead is the wrong scale entirely.
    if r <= 1.3: return 1.0
    if r <= 1.8: return 0.6
    if r <= 2.5: return 0.3
    return 0.1

def curvature_score(expected_curved, arrow):
    gen_curved = arrow["curve"]["curved"]
    if expected_curved == gen_curved: return 1.0
    # Wrong curvature — no partial credit
    return 0.0

def dash_score(expected, detected):
    if expected == detected: return 1.0
    if expected == "none" or detected == "none": return 0.0
    return 0.5

def match_arrow(arrow, from_label, to_label, shapes, reverse=False):
    """Match arrow to expected source/dest. If reverse, swap start/end."""
    if reverse:
        sl, _ = find_shape_at(*arrow["end"], shapes)
        el, _ = find_shape_at(*arrow["start"], shapes)
    else:
        sl, _ = find_shape_at(*arrow["start"], shapes)
        el, _ = find_shape_at(*arrow["end"], shapes)
    return {
        "src_match": 1.0 if sl == from_label else 0.0,
        "dst_match": 1.0 if el == to_label else 0.0,
        "from": sl, "to": el, "reversed": reverse
    }

def score_single_arrow(best, best_info, exp_color, exp_curved, gt_w, gen_shapes,
                       from_label, to_label, gt_sw=2.0, head_expected=True):
    """Score a single matched arrow across all metrics. Returns dict of metric scores.

    head_expected=False means the GT arrowhead tip is hidden under a shape in the
    ground-truth SVG.  In that case the model should NOT be penalized for omitting
    the arrowhead, so both head and head_size are awarded full credit regardless of
    whether the model included one.
    """
    src_sc  = best_info["src_match"]
    dst_sc  = best_info["dst_match"]
    if not head_expected:
        # Arrowhead not visible in GT → no penalty for omitting or including it.
        head_sc      = 1.0
        head_size_sc = 1.0
    else:
        head_sc      = 1.0 if best["has_head"] else 0.0
        head_size_sc = arrowhead_size_score(best, gt_w, gt_sw)
    curve_sc = curvature_score(exp_curved, best)
    color_sc = color_score(exp_color, best["color"])
    ovlp_sc  = overlap_score(best, gen_shapes, from_label, to_label)
    return {
        "source": src_sc, "dest": dst_sc, "head": head_sc,
        "head_size": head_size_sc, "curve": curve_sc,
        "color": color_sc, "overlap": ovlp_sc,
    }

# Visualization

def draw_annotations(svg_path, arrows, out_path, scores, shapes, used, n_gt, n_extra, n_missing, extra_penalty, composite):
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    cw, ch = canvas_size(root)

    g = ET.SubElement(root, "g", id="arrow-overlay", opacity="0.9")

    # Detected shape boxes (blue)
    for label, info in shapes.items():
        b = info["bounds"]
        r = ET.SubElement(g, "rect", x=str(b[0]), y=str(b[1]), width=str(b[2]-b[0]), height=str(b[3]-b[1]))
        r.set("fill", "none"); r.set("stroke", "#0088FF"); r.set("stroke-width", "2")
        r.set("stroke-dasharray", "8,4"); r.set("opacity", "0.7")
        t = ET.SubElement(g, "text", x=str(b[0]+3), y=str(b[1]-3))
        t.set("font-family", "monospace"); t.set("font-size", "9"); t.set("fill", "#0088FF")
        t.text = f"Det: {label[:12]}"

    # Arrows
    for i, arr in enumerate(arrows):
        sx, sy = arr["start"]
        ex, ey = arr["end"]
        mx, my = (sx+ex)/2, (sy+ey)/2

        is_extra = i not in used
        marker_color = "#999" if is_extra else "#FF00FF"

        c1 = ET.SubElement(g, "circle", cx=str(sx), cy=str(sy), r="5")
        c1.set("fill", "#00FF00"); c1.set("stroke", "#000"); c1.set("opacity", "0.8")
        c2 = ET.SubElement(g, "circle", cx=str(ex), cy=str(ey), r="5")
        c2.set("fill", "#FF0000"); c2.set("stroke", "#000"); c2.set("opacity", "0.8")

        bg = ET.SubElement(g, "circle", cx=str(mx), cy=str(my), r="12")
        bg.set("fill", "white"); bg.set("stroke", marker_color); bg.set("stroke-width", "2")
        num = ET.SubElement(g, "text", x=str(mx), y=str(my+4))
        num.set("font-family", "Arial"); num.set("font-size", "12"); num.set("font-weight", "bold")
        num.set("fill", marker_color); num.set("text-anchor", "middle")
        num.text = str(i)

    # Legend
    lg = ET.SubElement(g, "g", id="legend")
    legend_h = 50 + len(arrows)*16 + (30 if n_extra > 0 or n_missing > 0 else 0)
    lbg = ET.SubElement(lg, "rect", x="5", y="5", width=str(min(cw-10, 700)), height=str(legend_h))
    lbg.set("fill", "white"); lbg.set("stroke", "#333"); lbg.set("rx", "4"); lbg.set("opacity", "0.95")

    headers = ["#", "Expected", "Detected", "Src", "Dst", "Head", "Size", "Curve", "Ovlp", "Color", "Comp"]
    cols = [15, 35, 180, 320, 355, 390, 425, 465, 505, 545, 585]
    for j, h in enumerate(headers):
        t = ET.SubElement(lg, "text", x=str(cols[j]), y="22")
        t.set("font-family", "monospace"); t.set("font-size", "10"); t.set("font-weight", "bold")
        t.text = h

    for i, arr in enumerate(arrows):
        y = 40 + i*16
        sc = scores[i] if i < len(scores) else {}
        is_extra = i not in used
        t = ET.SubElement(lg, "text", x="15", y=str(y))
        t.set("font-family", "monospace"); t.set("font-size", "9")
        t.set("fill", "#999" if is_extra else "#FF00FF")
        t.text = str(i)
        if sc:
            t2 = ET.SubElement(lg, "text", x="35", y=str(y))
            t2.set("font-family", "monospace"); t2.set("font-size", "9")
            t2.text = f"{sc.get('exp_from','?')[:12]}->{sc.get('exp_to','?')[:12]}"
            t3 = ET.SubElement(lg, "text", x="180", y=str(y))
            t3.set("font-family", "monospace"); t3.set("font-size", "9")
            t3.text = f"{(sc.get('det_from') or '?')[:12]}->{(sc.get('det_to') or '?')[:12]}"
            vals = [sc.get(k, 0) for k in ["source", "dest", "head", "head_size", "curve", "overlap", "color", "composite"]]
            for j, v in enumerate(vals):
                t4 = ET.SubElement(lg, "text", x=str(cols[3+j]), y=str(y))
                t4.set("font-family", "monospace"); t4.set("font-size", "9")
                t4.set("fill", "#060" if v >= 0.8 else "#960" if v >= 0.5 else "#C00")
                t4.text = f"{v:.2f}"
        elif is_extra:
            t2 = ET.SubElement(lg, "text", x="35", y=str(y))
            t2.set("font-family", "monospace"); t2.set("font-size", "9"); t2.set("fill", "#999")
            sl, _ = find_shape_at(*arr["start"], shapes)
            el, _ = find_shape_at(*arr["end"], shapes)
            t2.text = f"[extra] {(sl or '?')[:10]}->{(el or '?')[:10]}"

    if n_missing > 0 or n_extra > 0:
        py = 40 + len(arrows)*16 + 5
        pt = ET.SubElement(lg, "text", x="15", y=str(py))
        pt.set("font-family", "monospace"); pt.set("font-size", "10"); pt.set("font-weight", "bold")
        pt.set("fill", "#C00")
        parts = [f"GT:{n_gt}", f"Det:{len(arrows)}"]
        if n_missing > 0:
            parts.append(f"Missing:{n_missing}(scored 0)")
        if n_extra > 0:
            parts.append(f"Extra:{n_extra}(x{1-extra_penalty:.2f})")
        parts.append(f"Composite:{composite:.3f}")
        pt.text = "  ".join(parts)

    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)

# Public API

_ARROW_METRICS = ["source", "dest", "head", "head_size", "curve", "color", "overlap"]
WEIGHTS = {k: 1.0 / len(_ARROW_METRICS) for k in _ARROW_METRICS}

def evaluate_arrows(sample_id, objects_dir, gt_svg_dir, model_output_dir, ann_dir=None):
    """Evaluate arrows for one sample (directory-based). Returns (scores_dict, report_string)."""
    objects_dir, gt_svg_dir, model_output_dir = Path(objects_dir), Path(gt_svg_dir), Path(model_output_dir)
    return evaluate_arrows_file(
        gen_svg=model_output_dir / f"{sample_id}.svg",
        gt_svg=gt_svg_dir / f"{sample_id}.svg",
        meta_json=objects_dir / f"{sample_id}.json",
        ann_path=Path(ann_dir) / f"{sample_id}_arrows.svg" if ann_dir else None,
        label=sample_id,
    )

def evaluate_arrows_file(gen_svg, gt_svg, meta_json, ann_path=None, label=None, ann_base_svg=None):
    """Evaluate arrows for one SVG file. Returns (scores_dict, report_string).

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
    gt_arrows = {k: v for k, v in meta["entities"].items() if v.get("type") == "arrow"}

    if not gt_arrows:
        return {"source": 1.0, "dest": 1.0, "head": 1.0, "head_size": 1.0,
                "curve": 1.0, "color": 1.0, "overlap": 1.0, "composite": 1.0}, "No arrows"

    gt_root = ET.parse(str(gt_svg)).getroot()
    # Pass GT shape bboxes so extract_gt_arrowheads can flag hidden arrowheads.
    gt_shape_bboxes = [v["bounds"] for v in gt_shapes.values()]
    gt_heads = extract_gt_arrowheads(gt_root, gt_shape_bboxes=gt_shape_bboxes)
    avg_gt_width = sum(h["width"] for h in gt_heads) / len(gt_heads) if gt_heads else 10

    gen_path = gen_svg
    gen_root = ET.parse(str(gen_path)).getroot()
    defs = parse_defs(gen_root)
    gen_shapes_list = extract_shapes(gen_root, defs)
    gen_shapes = {normalize_label(s["label"]): {"bounds": s["bounds"], "center": s["center"]} for s in gen_shapes_list}
    shape_bboxes = [s["bounds"] for s in gen_shapes_list]
    gen_arrows = extract_arrows(gen_root)
    gen_arrows = filter_arrows_inside_shapes(gen_arrows, shape_bboxes)

    cat_scores = {k: [] for k in WEIGHTS}
    arrow_scores = [None] * len(gen_arrows)
    used = set()

    shape_labels = list(gen_shapes.keys())[:5]
    report = [f"{'='*50}", f"ARROW EVAL: {sample_id}",
              f"Expected: {len(gt_arrows)} Detected: {len(gen_arrows)}",
              f"Shape labels for matching: {', '.join(shape_labels)}{'...' if len(gen_shapes)>5 else ''}", ""]

    # Sort numerically (arrow_0, arrow_1, ..., arrow_10) to match SVG document order.
    # sorted() gives lexicographic order (arrow_0, arrow_1, arrow_10, arrow_2, ...)
    # which misaligns gt_heads[] index with the SVG arrow position.
    def _arrow_key(k): return int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0
    gt_idx = 0
    for aid in sorted(gt_arrows, key=_arrow_key):
        arr = gt_arrows[aid]
        from_label = normalize_label(gt_shapes[arr["from"]]["label"])
        to_label = normalize_label(gt_shapes[arr["to"]]["label"])
        exp_color = arr.get("color", "#000")
        exp_curved = arr.get("style", "straight") == "curved"
        exp_dash = arr.get("dash", "none")

        report.append(f"--- {aid}: {from_label} -> {to_label} ---")
        report.append(f"  GT color={exp_color}  style={'curved' if exp_curved else 'straight'}  dash={exp_dash}")

        # Find best matching generated arrow
        best, best_score, best_idx, best_info = None, -1, -1, None
        for i, ga in enumerate(gen_arrows):
            if i in used: continue
            fwd = match_arrow(ga, from_label, to_label, gen_shapes, reverse=False)
            rev = match_arrow(ga, from_label, to_label, gen_shapes, reverse=True)
            if fwd["src_match"] + fwd["dst_match"] >= rev["src_match"] + rev["dst_match"]:
                info, score = fwd, fwd["src_match"] + fwd["dst_match"]
            else:
                info, score = rev, rev["src_match"] + rev["dst_match"]
            if score > best_score:
                best, best_score, best_idx, best_info = ga, score, i, info

        if not best or best_score == 0:
            report.append("  MATCH: NOT FOUND -- all scores 0.0")
            for k in cat_scores: cat_scores[k].append(0.0)
            gt_idx += 1
            continue

        used.add(best_idx)

        report.append(f"  GEN arrow[{best_idx}]  start=({best['start'][0]:.0f},{best['start'][1]:.0f})  end=({best['end'][0]:.0f},{best['end'][1]:.0f})")
        report.append(f"  GEN color={best['color']}  curved={best['curve']['curved']}  type={best['curve']['type']}  dash={best['dash']}  has_head={best['has_head']}")
        report.append(f"  GEN detected: {best_info.get('from','?')} -> {best_info.get('to','?')}  {'(reversed)' if best_info.get('reversed') else ''}")

        gt_w            = gt_heads[gt_idx]["width"]        if gt_idx < len(gt_heads) else avg_gt_width
        gt_sw           = gt_heads[gt_idx]["sw"]           if gt_idx < len(gt_heads) else 2.0
        head_expected   = gt_heads[gt_idx]["head_visible"] if gt_idx < len(gt_heads) else True
        gt_idx += 1   # advance per GT arrow (matched or not)
        scores = score_single_arrow(best, best_info, exp_color, exp_curved, gt_w, gen_shapes,
                                    from_label, to_label, gt_sw, head_expected)

        for k, v in scores.items():
            cat_scores[k].append(v)

        comp = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
        arrow_scores[best_idx] = {
            **scores, "composite": comp,
            "exp_from": from_label, "exp_to": to_label,
            "det_from": best_info["from"], "det_to": best_info["to"]
        }

        report.append(f"  SCORES:")
        report.append(f"    source:    {scores['source']:.3f}  (weight {WEIGHTS['source']:.0%})  expected={from_label} detected={best_info.get('from','?')}")
        report.append(f"    dest:      {scores['dest']:.3f}  (weight {WEIGHTS['dest']:.0%})  expected={to_label} detected={best_info.get('to','?')}")
        head_note = "hidden-in-GT" if not head_expected else f"has_head={best['has_head']}"
        report.append(f"    head:      {scores['head']:.3f}  (weight {WEIGHTS['head']:.0%})  {head_note} reversed={best_info.get('reversed',False)}")
        report.append(f"    head_size: {scores['head_size']:.3f}  (weight {WEIGHTS['head_size']:.0%})  gt_width={gt_w:.1f}")
        report.append(f"    curve:     {scores['curve']:.3f}  (weight {WEIGHTS['curve']:.0%})  expected={'curved' if exp_curved else 'straight'} detected={'curved' if best['curve']['curved'] else 'straight'}")
        report.append(f"    color:     {scores['color']:.3f}  (weight {WEIGHTS['color']:.0%})  GT={exp_color} GEN={best['color']}")
        report.append(f"    overlap:   {scores['overlap']:.3f}  (weight {WEIGHTS['overlap']:.0%})  endpoints inside shapes penalty")
        report.append(f"  COMPOSITE:   {comp:.3f}")
        gt_idx += 1

    # Missing / extra arrow accounting
    n_gt = len(gt_arrows)
    n_missing = sum(1 for v in cat_scores.get("source", []) if v == 0.0) - 0  # counted via 0.0 entries
    # Precise missing count: GT arrows that got no match
    n_matched = len(used)
    n_missing = n_gt - n_matched
    n_extra = len(gen_arrows) - n_matched
    extra_penalty = n_extra / (n_gt + n_extra) if (n_gt + n_extra) > 0 else 0.0

    if n_missing > 0:
        report.append("")
        report.append(f"--- MISSING ARROWS (in GT, not in output): {n_missing}/{n_gt} ---")
        report.append(f"  Penalty: each missing arrow scores 0.0 on all metrics (already included in averages above)")

    if n_extra > 0:
        report.append("")
        report.append(f"--- EXTRA ARROWS (not in GT): {n_extra} ---")
        for i, arr in enumerate(gen_arrows):
            if i not in used:
                sl, _ = find_shape_at(*arr["start"], gen_shapes)
                el, _ = find_shape_at(*arr["end"], gen_shapes)
                report.append(f"  arrow[{i}]  ({arr['start'][0]:.0f},{arr['start'][1]:.0f})->({arr['end'][0]:.0f},{arr['end'][1]:.0f})  {sl or '?'}->{el or '?'}")
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
    if n_missing > 0:
        report.append(f"MISSING ARROWS:  {n_missing}/{n_gt}  (scored 0.0, already in averages)")
    if n_extra > 0:
        report.append(f"RAW COMPOSITE:   {raw_composite:.3f}")
        report.append(f"EXTRA PENALTY:   -{extra_penalty:.3f}  ({n_extra} extra arrows / {n_gt + n_extra} total)")
    report.append(f"COMPOSITE: {composite:.3f}")

    if ann_path:
        ann_path = Path(ann_path)
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        base_svg = Path(ann_base_svg) if ann_base_svg else gen_path
        draw_annotations(base_svg, gen_arrows, ann_path,
                        arrow_scores, gen_shapes, used, n_gt, n_extra, n_missing, extra_penalty, composite)

    return {**avg, "extra_penalty": extra_penalty, "composite": composite}, "\n".join(report)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate arrows for one SVG pair.")
    ap.add_argument("gen_svg",    help="Generated (model output) SVG")
    ap.add_argument("gt_svg",     help="Ground-truth SVG")
    ap.add_argument("meta_json",  help="Ground-truth metadata JSON")
    ap.add_argument("--annotate", metavar="OUT_SVG", default=None,
                    help="Write annotated overlay SVG to this path")
    a = ap.parse_args()
    scores, report = evaluate_arrows_file(a.gen_svg, a.gt_svg, a.meta_json, ann_path=a.annotate)
    print(report)
    print(f"\nComposite: {scores['composite']:.3f}")
