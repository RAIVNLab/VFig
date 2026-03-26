import os
import json
import random
import time
import multiprocessing as mp
import traceback
import re

# ================================
# CONFIG
# ================================
SVG_DIR = "/path/to/svg_data"
OUTPUT_JSON = "finetune_data_xxx.json"
RENDER_DIR = "renders_xxx"

SAMPLE_LIMIT = None          # e.g. 20000
RENDER_TIMEOUT_SEC = 6       # kill per-SVG render if it hangs
PRINT_EVERY = 1000

os.makedirs(RENDER_DIR, exist_ok=True)

# ----- Instruction templates -----
TEMPLATES = [
    "Generate the SVG code for this image.",
    "Help me write the SVG code corresponding to this image.",
    "Please produce the SVG code to match this picture.",
    "Write the SVG code for the figure shown below.",
    "Output the SVG that reproduces this image."
]

# ================================
# NEW: Flatten RGBA -> RGB on white ONLY if transparency exists
# ================================
def _extract_svg_background_color(svg_text: str):
    """
    Extract background-color from the root <svg> style attribute, if present.
    Returns a CSS color string (e.g., "#F9EFF9") or None.
    """
    # Look specifically at the root <svg ... style="..."> to avoid matching
    # unrelated element styles.
    m = re.search(r"<svg[^>]*\sstyle=\"[^\"]*background-color\s*:\s*([^;\"\s]+)", svg_text, re.IGNORECASE)
    if not m:
        return None
    color = m.group(1).strip()
    if color.lower() in ("transparent", "none"):
        return None
    # Simple filter for rgba(..., 0) to keep transparency intent.
    if color.lower().startswith("rgba") and color.rstrip(")").endswith(",0"):
        return None
    return color

def _flatten_if_transparent(png_path: str) -> bool:
    """
    CHANGED/NEW:
    - If the rendered PNG has alpha and contains any transparent pixels,
      flatten it onto a white background and save as RGB.
    - If fully opaque (alpha all 255), keep original pixels.
    """
    from PIL import Image

    img = Image.open(png_path)

    # If no alpha channel at all -> nothing to do
    if img.mode in ("RGB", "L"):
        return False

    # Convert to RGBA to inspect alpha reliably
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = img.getchannel("A")
    # If alpha is fully opaque everywhere -> keep original (preserve bg)
    if alpha.getextrema() == (255, 255):
        # CHANGED: still save as RGB for consistency (optional)
        # If you want to keep as RGBA, comment the next 2 lines.
        img.convert("RGB").save(png_path)
        return False

    # Otherwise, flatten onto white
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, (0, 0), img)  # use alpha as mask
    bg.save(png_path)
    return True


# ================================
# SAFE RENDER (SUBPROCESS + TIMEOUT)
# ================================
def _render_worker(svg_text: str, out_png: str, q: mp.Queue):
    try:
        import cairosvg

        # (unchanged) render
        bg_color = _extract_svg_background_color(svg_text)
        if bg_color:
            cairosvg.svg2png(
                bytestring=svg_text.encode("utf-8"),
                write_to=out_png,
                background_color=bg_color
            )
        else:
        cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=out_png)

        # CHANGED: only flatten if transparency exists
        flattened = _flatten_if_transparent(out_png)

        q.put(("ok", {"flattened": flattened}))
    except Exception as e:
        q.put(("err", f"{e}\n{traceback.format_exc()}"))

def safe_render(svg_text: str, out_png: str, timeout_sec: int):
    q = mp.Queue(maxsize=1)
    p = mp.Process(
        target=_render_worker,
        args=(svg_text, out_png, q),
        daemon=True
    )
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        p.kill()
        p.join()
        return False, f"timeout>{timeout_sec}s (likely infinite recursion / pathological SVG)", None

    try:
        status, detail = q.get_nowait()
    except Exception:
        return False, f"worker exited without status (exitcode={p.exitcode})", None

    if status != "ok":
        return False, detail, None

    return True, None, detail

# ================================
# MAIN PIPELINE
# ================================
def main():
    all_svgs = sorted(f for f in os.listdir(SVG_DIR) if f.endswith(".svg"))
    if SAMPLE_LIMIT is not None:
        all_svgs = all_svgs[:int(SAMPLE_LIMIT)]

    total = len(all_svgs)
    print(f"📂 Found {total} SVG files.", flush=True)

    final_data = []
    start_time = time.time()

    reused = 0
    rendered = 0
    skipped = 0
    flattened = 0
    flattened_examples = []

    for idx, fname in enumerate(all_svgs):
        svg_path = os.path.join(SVG_DIR, fname)

        # ---- read svg ----
        try:
            with open(svg_path, "r", encoding="utf-8", errors="replace") as f:
                svg_text = (
                    f.read()
                    .replace('\\"', '"')
                    .replace("\\n", "\n")
                )
        except Exception as e:
            skipped += 1
            print(f"⚠️ Skipped {fname} due to read error: {e}", flush=True)
            continue

        # ---- render (resume-safe) ----
        img_filename = f"{idx:05d}_{fname[:-4]}.png"
        img_path = os.path.join(RENDER_DIR, img_filename)

        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            reused += 1
        else:
            ok, reason, info = safe_render(svg_text, img_path, RENDER_TIMEOUT_SEC)
            if not ok:
                skipped += 1
                print(f"⚠️ Skipped {fname} due to render error: {reason}", flush=True)
                try:
                    if os.path.exists(img_path) and os.path.getsize(img_path) == 0:
                        os.remove(img_path)
                except Exception:
                    pass
                continue
            rendered += 1
            if info and info.get("flattened"):
                flattened += 1
                if len(flattened_examples) < 5:
                    flattened_examples.append(img_path)

        # ---- record ----
        final_data.append({
            "messages": [
                {"role": "user", "content": f"<image>\n{random.choice(TEMPLATES)}"},
                {"role": "assistant", "content": svg_text}
            ],
            "images": [img_path]
        })

        # ---- progress ----
        if (idx + 1) % PRINT_EVERY == 0 or (idx + 1) == total:
            elapsed = time.time() - start_time
            print(
                f"✅ Processed {idx + 1}/{total} "
                f"({elapsed:.1f}s) | ok={len(final_data)} "
                f"reused={reused} new={rendered} skipped={skipped}",
                flush=True
            )

    # ---- save JSON ----
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print("\n🎉 Done!", flush=True)
    print(f"👉 Saved {len(final_data)} records to {OUTPUT_JSON}", flush=True)
    print(
        f"🖼️ Images in {RENDER_DIR}/ "
        f"(reused={reused}, new={rendered}, skipped={skipped})",
        flush=True
    )
    print(
        f"🔍 Transparent images flattened to white: {flattened}",
        flush=True
    )
    if flattened_examples:
        print("🧾 Examples (up to 5):", flush=True)
        for ex in flattened_examples:
            print(f"  - {ex}", flush=True)

# ================================
# RUN
# ================================
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
