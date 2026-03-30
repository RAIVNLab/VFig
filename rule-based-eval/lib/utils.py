#!/usr/bin/env python3
"""Shared helpers for shape and arrow evaluation."""

import sys, math, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Maximum L2 distance in integer RGB space: sqrt((255^2)*3) = 255*sqrt(3).
# Used to normalise Euclidean RGB distance to [0, 1] with a principled upper
# bound — not an empirical magic number.  Any two colours score >= 0 because
# the actual distance is always <= this theoretical maximum.
_RGB_MAX_DIST = 255.0 * math.sqrt(3)  # ≈ 441.67


def strip_ns(tag):
    return tag.split("}", 1)[1] if "}" in tag else tag

def safe_float(s, default=0.0):
    if not s or "%" in s: return default
    try: return float(s)
    except: return default

def find_nums(s):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s or "")]

def parse_translate(t):
    m = re.search(r"translate\(\s*([-\d.]+)[,\s]+([-\d.]+)", t or "")
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)

def normalize_label(text):
    return " ".join((text or "").lower().split())

# CSS named colours → hex (common subset)
_NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "lime": "#00ff00",
    "blue": "#0000ff", "yellow": "#ffff00", "cyan": "#00ffff", "aqua": "#00ffff",
    "magenta": "#ff00ff", "fuchsia": "#ff00ff", "silver": "#c0c0c0", "gray": "#808080",
    "grey": "#808080", "maroon": "#800000", "olive": "#808000", "green": "#008000",
    "purple": "#800080", "teal": "#008080", "navy": "#000080", "orange": "#ffa500",
    "coral": "#ff7f50", "salmon": "#fa8072", "tomato": "#ff6347", "gold": "#ffd700",
    "khaki": "#f0e68c", "violet": "#ee82ee", "indigo": "#4b0082", "brown": "#a52a2a",
    "tan": "#d2b48c", "beige": "#f5f5dc", "ivory": "#fffff0", "lavender": "#e6e6fa",
    "turquoise": "#40e0d0", "crimson": "#dc143c", "darkred": "#8b0000",
    "darkgreen": "#006400", "darkblue": "#00008b", "darkorange": "#ff8c00",
    "darkgray": "#a9a9a9", "darkgrey": "#a9a9a9", "lightgray": "#d3d3d3",
    "lightgrey": "#d3d3d3", "lightblue": "#add8e6", "lightyellow": "#ffffe0",
    "lightgreen": "#90ee90", "pink": "#ffc0cb", "hotpink": "#ff69b4",
    "deepskyblue": "#00bfff", "dodgerblue": "#1e90ff", "royalblue": "#4169e1",
    "steelblue": "#4682b4", "slategray": "#708090", "slategrey": "#708090",
    "dimgray": "#696969", "dimgrey": "#696969", "whitesmoke": "#f5f5f5",
    "snow": "#fffafa", "mintcream": "#f5fffa", "honeydew": "#f0fff0",
    "aliceblue": "#f0f8ff", "ghostwhite": "#f8f8ff", "floralwhite": "#fffaf0",
    "linen": "#faf0e6", "seashell": "#fff5ee", "oldlace": "#fdf5e6",
    "transparent": "none",
}


def hex_to_rgb(h):
    h = (h or "").lstrip("#").lower()
    if len(h) == 3: h = h[0]*2 + h[1]*2 + h[2]*2
    if len(h) != 6: return None
    try: return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except: return None


def _parse_color(c):
    """Normalise a CSS colour string to an (r,g,b) tuple or None.

    Handles:  #rrggbb  #rgb  rgb(r,g,b)  rgb(r%,g%,b%)  CSS named colours
    """
    c = (c or "").strip().lower()
    if not c or c in ("none", ""):
        return None
    # Named colour
    if c in _NAMED_COLORS:
        c = _NAMED_COLORS[c]
        if c == "none":
            return None
    # rgb(...) / rgba(...)
    m = re.match(r"rgba?\(\s*([\d.]+%?)\s*,\s*([\d.]+%?)\s*,\s*([\d.]+%?)", c)
    if m:
        def _ch(s):
            s = s.strip()
            v = float(s.rstrip("%"))
            return int(v * 2.55) if s.endswith("%") else int(round(v))
        try:
            return (_ch(m.group(1)), _ch(m.group(2)), _ch(m.group(3)))
        except Exception:
            return None
    return hex_to_rgb(c)


def color_score(c1, c2):
    """RGB Euclidean distance score (0–1).

    Handles hex (#rrggbb / #rgb), CSS named colours, rgb() / rgba(), and url() refs.
    """
    c1, c2 = (c1 or "").strip().lower(), (c2 or "").strip().lower()
    # url() pattern references
    if c1.startswith("url(") and c2.startswith("url("):
        return 0.5  # both are pattern/gradient refs — neutral
    if c1.startswith("url(") or c2.startswith("url("):
        # One is a pattern ref, other is a colour — can't compare reliably
        return 0.3
    # Resolve named colours before none-check (transparent -> "none")
    c1 = _NAMED_COLORS.get(c1, c1)
    c2 = _NAMED_COLORS.get(c2, c2)
    if c1 in ("none", "") and c2 in ("none", ""):
        return 1.0
    if c1 in ("none", "") or c2 in ("none", ""):
        return 0.0
    rgb1, rgb2 = _parse_color(c1), _parse_color(c2)
    if not rgb1 or not rgb2:
        return 0.0
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
    return max(0.0, 1.0 - dist / _RGB_MAX_DIST)

# Zero-score dicts used when a sample is missing or errored during evaluation.
ZERO_SHAPE_SCORES = {
    "label": 0., "type": 0., "fill_color": 0., "fill_style": 0.,
    "stroke_color": 0., "border_style": 0., "position": 0.,
    "font": 0., "aspect_ratio": 0., "extra_penalty": 0., "composite": 0.,
}
ZERO_ARROW_SCORES = {
    "source": 0., "dest": 0., "head": 0., "head_size": 0.,
    "curve": 0., "color": 0., "overlap": 0., "extra_penalty": 0., "composite": 0.,
}


def canvas_size(root):
    vb = root.get("viewBox")
    if vb:
        p = vb.split()
        return float(p[2]), float(p[3])
    return safe_float(root.get("width"), 800), safe_float(root.get("height"), 600)
