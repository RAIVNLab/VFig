import re
import io
import os
import json
import time
import random
import numpy as np
import cv2
import cairosvg
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from func_timeout import func_timeout, FunctionTimedOut

from google import genai
from google.genai import types, errors
# Set GEMINI_API_KEY as an environment variable before running:
#   export GEMINI_API_KEY=<your_api_key>

SVG_RE = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)


def _extract_svg(text: str):
    if not text: return None
    m = SVG_RE.search(text)
    return m.group(0) if m else None


def _safe_load_on_white(img):
    """Composites the image onto a white background to handle transparency."""
    if img.mode == "RGB":
        return img
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    bg = Image.new('RGB', img.size, (255, 255, 255))
    bg.paste(img, (0, 0), img)
    return bg


def _render_png(svg_text: str, size):
    W, H = size
    try:
        png_bytes = func_timeout(
            30.0,
            cairosvg.svg2png,
            kwargs={
                "bytestring": svg_text.encode("utf-8"),
                "output_width": int(W),
                "output_height": int(H),
            }
        )
        img = Image.open(io.BytesIO(png_bytes))
        return _safe_load_on_white(img)

    except FunctionTimedOut:
        print(f"Render Error: CairoSVG Timed Out (>30s)", flush=True)
        return None

    except Exception:
        return None


def _load_gt_png(png_path: str):
    img = Image.open(png_path)
    img = _safe_load_on_white(img)
    return img, img.size


def _call_gemini_reward(img_pred_pil, img_gt_pil):
    worker_id = f"[PID:{os.getpid()}]"

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=60000)
    )

    prompt = """
    System Role: You are a meticulous visual quality auditor for scientific diagrams.

    Task: Compare the Candidate (Generated Image) against the Ground Truth (Reference Image).
    Focus ONLY on visual fidelity. Do not evaluate the underlying code.

    Scoring Components [0.0 to 1.0]:
    1. Presence: Are all visual elements (labels, boxes, shapes, text) present?
    2. Layout: Is the placement, spacing, and alignment accurate relative to the reference?
    3. Connectivity: Do arrows and lines connect the correct source/target elements?
    4. Text & Details: Is the text legible and correct? Are small details (arrowheads, dashed lines) preserved?

    Output exactly in this JSON format:
    {"presence": float, "layout": float, "connectivity": float, "details": float, "reasoning": "string"}
    """

    contents = [
        "GROUND TRUTH IMAGE:", img_gt_pil,
        "CANDIDATE IMAGE:",    img_pred_pil,
        prompt
    ]

    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=16384,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_level="high"
                    )
                )
            )

            thoughts = ""
            final_json_text = ""

            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if getattr(part, 'thought', False) is True:
                        thoughts += part.text
                    else:
                        final_json_text += part.text

            if not final_json_text and response.text:
                final_json_text = response.text or ""

            if not final_json_text.strip():
                finish_reason = "Unknown"
                if hasattr(response.candidates[0], 'finish_reason'):
                    finish_reason = str(response.candidates[0].finish_reason)
                raise ValueError(f"Empty Response (Blocked: {finish_reason})")

            try:
                clean_text = final_json_text.replace("```json", "").replace("```", "").strip()
                last_brace_index = clean_text.rfind("}")
                if last_brace_index != -1:
                    clean_text = clean_text[:last_brace_index + 1]

                data = json.loads(clean_text)

                if isinstance(data, list):
                    if len(data) > 0:
                        data = data[0]
                    else:
                        raise ValueError("Model returned an empty list")

            except json.JSONDecodeError:
                raise ValueError("Invalid JSON received")

            if not thoughts and "reasoning" in data:
                thoughts = data["reasoning"]

            weights = {
                'presence': 0.25,
                'layout': 0.25,
                'connectivity': 0.25,
                'details': 0.25
            }

            weighted_score = (
                data.get('presence', 0) * weights['presence'] +
                data.get('layout', 0) * weights['layout'] +
                data.get('connectivity', 0) * weights['connectivity'] +
                data.get('details', 0) * weights['details']
            )

            return weighted_score, data, thoughts

        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            else:
                print(f"Gemini Client Error (Fatal): {e}")
                return 0.0, {}, f"Fatal Error: {str(e)}"

        except errors.ServerError as e:
            sleep_time = base_delay * (2 ** attempt) + random.uniform(1, 3)
            print(f"Server Overloaded. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)

        except Exception as e:
            if "deadline" in str(e).lower() or "timeout" in str(e).lower():
                print(f"{worker_id} TIMEOUT on Attempt {attempt+1}. Retrying...", flush=True)
                time.sleep(1)
            else:
                if attempt == max_retries - 1:
                    return 0.0, {}, f"Error: {str(e)}"
                time.sleep(2)

    return 0.0, {}, "Failed after max retries."


def compute_score(solution_str: str, ground_truth, method="rule", format="score", extra_info=None, **kwargs):
    pred_svg_code = _extract_svg(solution_str)

    if not pred_svg_code:
        return {
            "score": 0.0, "presence": 0.0, "layout": 0.0,
            "connectivity": 0.0, "details": 0.0,
            "thoughts": "Failed to extract SVG tags."
        }

    gt_png_path = ""
    if isinstance(ground_truth, dict):
        gt_png_path = ground_truth.get('ground_truth', '')
    else:
        gt_png_path = str(ground_truth)

    try:
        img_gt_L, gt_size = _load_gt_png(gt_png_path)
        img_pred_L = _render_png(pred_svg_code, gt_size)

        if img_pred_L is None:
            return {
                "score": 0.0, "presence": 0.0, "layout": 0.0,
                "connectivity": 0.0, "details": 0.0,
                "thoughts": "Rendering Failed (Syntax Error)"
            }

        img_gt_rgb = img_gt_L.convert("RGB")
        img_pred_rgb = img_pred_L.convert("RGB")

        gemini_avg, details, thoughts = _call_gemini_reward(
            img_pred_pil=img_pred_rgb,
            img_gt_pil=img_gt_rgb
        )

        return {
            "score": float(gemini_avg),
            "presence": float(details.get('presence', 0)),
            "layout": float(details.get('layout', 0)),
            "connectivity": float(details.get('connectivity', 0)),
            "details": float(details.get('details', 0)),
            "thoughts": thoughts
        }

    except Exception as e:
        return {
            "score": 0.0, "presence": 0.0, "layout": 0.0,
            "connectivity": 0.0, "details": 0.0,
            "thoughts": f"Exception: {str(e)}"
        }


if __name__ == '__main__':
    def load_file_context(path):
        if not os.path.exists(path): return "<svg></svg>"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    svg_path = "data/good_png_svg/svg/simple_svg_off.svg"
    png_path = "data/good_png_svg/png/simple_svg_color.png"

    svg_pred = load_file_context(svg_path)

    if os.path.exists(svg_path) and os.path.exists(png_path):
        score_dict = compute_score(solution_str=svg_pred, ground_truth=png_path)
        print(json.dumps(score_dict, indent=2))
    else:
        print("Import check passed. (Test files not found, skipping local run)")
