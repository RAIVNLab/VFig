#!/usr/bin/env python3
"""
SVG->PNG evaluation on benchmark_data/original_img (reference PNGs) vs
benchmark_data/rendered_svg/mllm (generated SVGs).

Metrics (computed on successfully rendered pairs only):
- SSIM
- LPIPS
- DINO cosine
- CLIP image cosine
- SigLIP image cosine
- Gemini 3 Flash Score
- GPT-5.2 Score

Outputs (written under benchmark_data/eval_results/):
- mllm__per_image.json  (top-level stats + per-image records)
- mllm__summary.md      (dataset-level summary in Markdown)
"""

import argparse
import json
import multiprocessing as mp
import signal
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import io
import base64

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

from skimage.metrics import structural_similarity as ssim_fn
import lpips
import time
import os

# --- API Imports ---
from google import genai
from google.genai import types
from google.api_core import exceptions
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
    SiglipModel,
    SiglipProcessor,
)

# ==========================================================
# ====================== CONFIG ============================
# ==========================================================

# API Keys
# Set GEMINI_API_KEY in your environment before running
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set OPENAI_API_KEY in your environment before running

BENCHMARK_ROOT = Path("xxx")
ORIGINAL_IMG_ROOT = BENCHMARK_ROOT / "xxx"

OUTPUT_ROOT = Path("xxx")
EVAL_OUT_ROOT = OUTPUT_ROOT / "xxx"
RENDER_CACHE_ROOT = OUTPUT_ROOT / "xxx"

MODEL_NAME = "xxx"
MODEL_SVG_ROOT = Path("xxx")

def _gen_svg_dirname(base: str) -> str:
    """
    Return the actual generated-SVG subdirectory name.
    VTracer outputs SVGs without a white-bg rendering step, so its dirs use
    the plain names; the rendering pipeline already flattens transparent
    backgrounds to white at eval time.  All other models use the _white_bg
    directories produced by the white-background inference pipeline.
    """
    if MODEL_NAME in ("Vtracer", "Vtracer_white"):
        return base  # e.g. "arxiv_p2f_rendered_svg_397"
    return f"{base}_white_bg"  # e.g. "arxiv_p2f_rendered_svg_397_white_bg"

DATASETS = {
    "arxiv_p2f": {
        "ref_img_dir": ORIGINAL_IMG_ROOT / "arxiv_p2f_img_eval_397_white_bg",
        "gen_svg_dirname": _gen_svg_dirname("arxiv_p2f_rendered_svg_397"),
    },
    "molmo": {
        "ref_img_dir": ORIGINAL_IMG_ROOT / "molmo_img_eval_499_white_bg",
        "gen_svg_dirname": _gen_svg_dirname("molmo_rendered_svg_499"),
    },
    "starvector": {
        "ref_img_dir": ORIGINAL_IMG_ROOT / "starvector_img_eval_440_white_bg",
        "gen_svg_dirname": _gen_svg_dirname("starvector_rendered_svg_440"),
    },
}

# Embedding backbones
CLIP_CKPT = "openai/clip-vit-base-patch32"
DINO_CKPT = "facebook/dinov2-base"
SIGLIP_CKPT = "google/siglip2-so400m-patch14-384"


# ==========================================================
# ================= SVG -> PNG rendering ===================
# ==========================================================

def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _flatten_png_alpha_to_white(png_path: Path) -> None:
    """
    Ensure cached PNGs are RGB on white background if they contain transparency.
    This mutates the cached file in-place.
    """
    with Image.open(png_path) as img:
        if img.mode == "RGB":
            return
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        alpha = img.getchannel("A")
        if alpha.getextrema() == (255, 255):
            # Fully opaque; keep deterministic RGB cache format.
            img.convert("RGB").save(png_path)
            return

        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, (0, 0), img)
        bg.save(png_path)

def _cairosvg_render_worker(svg_bytes: bytes, png_path: str, size_px: Optional[int], q: "mp.Queue") -> None:
    try:
        import cairosvg  # type: ignore

        kwargs: Dict[str, Any] = {}
        if size_px is not None:
            kwargs["output_width"] = int(size_px)
            kwargs["output_height"] = int(size_px)
        cairosvg.svg2png(bytestring=svg_bytes, write_to=str(png_path), **kwargs)
        q.put({"ok": True})
    except Exception as e:
        q.put({"ok": False, "err": f"{type(e).__name__}: {e}"})

def _render_svg_to_png_inprocess(svg_path: Path, png_path: Path, size_px: Optional[int], timeout_sec: int) -> Tuple[bool, Optional[str]]:
    try:
        import cairosvg  # type: ignore

        svg_bytes = svg_path.read_bytes()
        kwargs: Dict[str, Any] = {}
        if size_px is not None:
            kwargs["output_width"] = int(size_px)
            kwargs["output_height"] = int(size_px)

        def _alarm_handler(_signum, _frame):  # type: ignore[no-untyped-def]
            raise TimeoutError("render timeout")

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        try:
            if timeout_sec and timeout_sec > 0:
                signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
            cairosvg.svg2png(bytestring=svg_bytes, write_to=str(png_path), **kwargs)
        finally:
            if timeout_sec and timeout_sec > 0:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)

        ok = png_path.exists() and png_path.stat().st_size > 0
        return (True, None) if ok else (False, "cairosvg_failed")
    except TimeoutError:
        return False, "timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def render_svg_to_png(
    svg_path: Path,
    png_path: Path,
    size_px: Optional[int] = None,
    timeout_sec: int = 30,
) -> Tuple[bool, Optional[str]]:
    png_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ok, err = _render_svg_to_png_inprocess(svg_path, png_path, size_px=size_px, timeout_sec=int(timeout_sec))
        if ok:
            return True, None
        if err == "timeout":
            return False, "timeout"
    except Exception:
        pass

    try:
        import cairosvg  # type: ignore  # noqa: F401
        ctx = mp.get_context("spawn")
        q: "mp.Queue" = ctx.Queue(maxsize=1)
        svg_bytes = svg_path.read_bytes()
        p = ctx.Process(target=_cairosvg_render_worker, args=(svg_bytes, str(png_path), size_px, q))
        p.start()
        p.join(timeout=max(1, int(timeout_sec)))
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
            return False, "timeout"
        try:
            msg = q.get_nowait()
        except Exception:
            msg = None
        ok = png_path.exists() and png_path.stat().st_size > 0
        if ok:
            _flatten_png_alpha_to_white(png_path)
            return True, None
        err = None
        if isinstance(msg, dict):
            err = msg.get("err")
        return False, err or "cairosvg_failed"
    except Exception:
        pass

    rsvg = _which("rsvg-convert")
    if rsvg:
        cmd = [rsvg, str(svg_path), "-o", str(png_path)]
        if size_px is not None:
            cmd += ["-w", str(int(size_px)), "-h", str(int(size_px))]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok = png_path.exists() and png_path.stat().st_size > 0
            if ok:
                _flatten_png_alpha_to_white(png_path)
            return (ok, None) if ok else (False, "rsvg_failed")
        except Exception:
            pass

    inkscape = _which("inkscape")
    if inkscape:
        cmd = [inkscape, str(svg_path), "--export-type=png", f"--export-filename={png_path}"]
        if size_px is not None:
            cmd += [f"--export-width={int(size_px)}", f"--export-height={int(size_px)}"]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok = png_path.exists() and png_path.stat().st_size > 0
            if ok:
                _flatten_png_alpha_to_white(png_path)
            return (ok, None) if ok else (False, "inkscape_failed")
        except Exception:
            pass

    return False, "no_svg_renderer_available"


# ==========================================================
# ===================== Image utils =========================
# ==========================================================
def _safe_load_on_white(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, (0, 0), img)
    return bg


def load_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    return _safe_load_on_white(img)

def resize_to_match(img: Image.Image, ref: Image.Image) -> Image.Image:
    if img.size == ref.size:
        return img
    return img.resize(ref.size, Image.BICUBIC)

def pil_to_np_uint8(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)

def compute_ssim(ref_img: Image.Image, gen_img: Image.Image) -> float:
    ref = pil_to_np_uint8(ref_img)
    gen = pil_to_np_uint8(gen_img)
    return float(ssim_fn(ref, gen, channel_axis=2, data_range=255))

def torch_image_for_lpips(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


# ==========================================================
# =================== Embedding models ======================
# ==========================================================

@dataclass
class Bundle:
    lpips_model: lpips.LPIPS
    clip_model: CLIPModel
    clip_proc: CLIPProcessor
    dino_model: torch.nn.Module
    dino_proc: AutoImageProcessor
    siglip_model: SiglipModel
    siglip_proc: SiglipProcessor
    device: torch.device

def load_bundle(device: torch.device) -> Bundle:
    lp = lpips.LPIPS(net="alex").to(device).eval()

    clip_model = CLIPModel.from_pretrained(CLIP_CKPT).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_CKPT)

    dino_model = AutoModel.from_pretrained(DINO_CKPT).to(device).eval()
    dino_proc = AutoImageProcessor.from_pretrained(DINO_CKPT)

    siglip_model = SiglipModel.from_pretrained(SIGLIP_CKPT).to(device).eval()
    siglip_proc = SiglipProcessor.from_pretrained(SIGLIP_CKPT)

    return Bundle(
        lpips_model=lp,
        clip_model=clip_model,
        clip_proc=clip_proc,
        dino_model=dino_model,
        dino_proc=dino_proc,
        siglip_model=siglip_model,
        siglip_proc=siglip_proc,
        device=device,
    )

@torch.no_grad()
def embed_clip(b: Bundle, img: Image.Image) -> torch.Tensor:
    inputs = b.clip_proc(images=[img], return_tensors="pt").to(b.device)
    feat = b.clip_model.get_image_features(**inputs)
    return F.normalize(feat, p=2, dim=-1)[0]

@torch.no_grad()
def embed_siglip(b: Bundle, img: Image.Image) -> torch.Tensor:
    inputs = b.siglip_proc(images=[img], return_tensors="pt").to(b.device)
    feat = b.siglip_model.get_image_features(**inputs)
    return F.normalize(feat, p=2, dim=-1)[0]

@torch.no_grad()
def embed_dino(b: Bundle, img: Image.Image) -> torch.Tensor:
    inputs = b.dino_proc(images=[img], return_tensors="pt").to(b.device)
    out = b.dino_model(**inputs)

    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat = out.pooler_output
    else:
        h = out.last_hidden_state
        feat = h[:, 1:, :].mean(dim=1) if h.shape[1] > 1 else h.mean(dim=1)

    return F.normalize(feat, p=2, dim=-1)[0]

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


# ==========================================================
# =================== File matching =========================
# ==========================================================

def list_ref_images(folder: Path) -> Dict[str, Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    m: Dict[str, Path] = {}
    if not folder.exists():
        return m
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            m[p.stem] = p
    return dict(sorted(m.items(), key=lambda x: x[0]))

def list_svgs(folder: Path) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    if not folder.exists():
        return m
    model_suffixes = {MODEL_NAME, MODEL_NAME.lower()}
    model_name_l = MODEL_NAME.lower()
    # Some pipelines name outputs with base model tags like "_gpt5_2"
    # while MODEL_NAME may be "gpt5_2_white". Accept both for matching.
    if model_name_l.endswith("_white"):
        base = MODEL_NAME[: -len("_white")]
        model_suffixes.add(base)
        model_suffixes.add(base.lower())
    # Gemini outputs are often saved with family-level suffixes like "_gemini3.svg"
    # even when MODEL_NAME is "gemini3_pro_white" or "gemini3_flash_white".
    if model_name_l.startswith("gemini3"):
        model_suffixes.update({"gemini3", "gemini_3", "gemini-3"})
        if "pro" in model_name_l:
            model_suffixes.update({"gemini3_pro", "gemini3pro"})
        if "flash" in model_name_l:
            model_suffixes.update({"gemini3_flash", "gemini3flash"})
    # Normalize aliases to handle naming variants like
    # "_gemini3flash", "_gemini3_flash", "_gpt5_2", etc.
    expanded_suffixes: set[str] = set()
    for s in model_suffixes:
        t = s.lower()
        expanded_suffixes.add(t)
        expanded_suffixes.add(t.replace("-", "_"))
        expanded_suffixes.add(t.replace("-", ""))
        expanded_suffixes.add(t.replace("_", ""))
        if t.endswith("_white"):
            base = t[: -len("_white")]
            base_compact = base.replace("_", "").replace("-", "")
            expanded_suffixes.add(base_compact + "_white")
    for p in folder.rglob("*.svg"):
        stem = p.stem
        suffixes = [f"_{s}" for s in sorted(expanded_suffixes, key=len, reverse=True)] + ["_adapter"]
        changed = True
        while changed:
            changed = False
            for suf in suffixes:
                if not suf:
                    continue
                if stem.lower().endswith(suf.lower()):
                    stem = stem[: -len(suf)]
                    changed = True
        m[stem] = p
    return dict(sorted(m.items(), key=lambda x: x[0]))


# ==========================================================
# ====================== Judges ============================
# ==========================================================

class GeminiJudge:
    def __init__(self, model_id: str = "gemini-3-flash-preview"):
        if not GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY not found. Gemini scoring will be skipped.")
            self.client = None
        else:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_id = model_id

    def _construct_prompt(self) -> str:
        return """
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

    def evaluate(self, ref_img: Image.Image, gen_img: Image.Image) -> Dict[str, float]:
        if self.client is None:
            return {k: np.nan for k in ["score", "presence", "layout", "connectivity", "details"]}

        ref_resized = ref_img.resize((512, 512))
        gen_resized = gen_img.resize((512, 512))

        contents = [
            "GROUND TRUTH IMAGE:", ref_resized,
            "CANDIDATE IMAGE:",    gen_resized,
            self._construct_prompt()
        ]

        max_retries = 5
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )

                text_resp = response.text.strip() if response.text else ""
                
                if not text_resp:
                    reason = "Unknown"
                    if response.candidates and response.candidates[0].finish_reason:
                        reason = str(response.candidates[0].finish_reason)
                    raise ValueError(f"Empty Response (Blocked: {reason})")

                try:
                    data = json.loads(text_resp)
                except json.JSONDecodeError:
                    clean_text = text_resp.replace("```json", "").replace("```", "").strip()
                    last_brace_index = clean_text.rfind("}")
                    if last_brace_index != -1:
                        clean_text = clean_text[:last_brace_index + 1]
                    data = json.loads(clean_text)

                if isinstance(data, list):
                    if len(data) > 0:
                        data = data[0]
                    else:
                        raise ValueError("Model returned an empty list")

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

                return {
                    "score": float(weighted_score),
                    "presence": float(data.get("presence", 0)),
                    "layout": float(data.get("layout", 0)),
                    "connectivity": float(data.get("connectivity", 0)),
                    "details": float(data.get("details", 0)),
                    "reasoning": data.get("reasoning", "")
                }

            except (exceptions.ServiceUnavailable, exceptions.InternalServerError, exceptions.ResourceExhausted) as e:
                sleep_time = base_delay * (2 ** attempt) + np.random.uniform(0, 1)
                print(f"Gemini API {type(e).__name__}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            
            except Exception as e:
                print(f"Gemini Eval Error (Attempt {attempt}): {e}")
                if attempt == max_retries - 1:
                    break
                time.sleep(2)

        return {k: np.nan for k in ["score", "presence", "layout", "connectivity", "details"]}


class GPTJudge:
    def __init__(self, model_id: str = "gpt-5.2"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not found. GPT scoring will be skipped.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        self.model_id = model_id

    def _construct_prompt(self) -> str:
        return """
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

    def _encode_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def evaluate(self, ref_img: Image.Image, gen_img: Image.Image) -> Dict[str, float]:
        if self.client is None:
            return {k: np.nan for k in ["score", "presence", "layout", "connectivity", "details"]}

        ref_resized = ref_img.resize((512, 512))
        gen_resized = gen_img.resize((512, 512))

        ref_b64 = self._encode_image(ref_resized)
        gen_b64 = self._encode_image(gen_resized)

        messages = [
            {
                "role": "system",
                "content": self._construct_prompt()
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "GROUND TRUTH IMAGE:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{ref_b64}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": "CANDIDATE IMAGE:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{gen_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        max_retries = 5
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                text_resp = response.choices[0].message.content
                if not text_resp:
                    raise ValueError("Empty response from OpenAI")

                try:
                    data = json.loads(text_resp)
                except json.JSONDecodeError:
                    clean_text = text_resp.replace("```json", "").replace("```", "").strip()
                    last_brace_index = clean_text.rfind("}")
                    if last_brace_index != -1:
                        clean_text = clean_text[:last_brace_index + 1]
                    data = json.loads(clean_text)

                if isinstance(data, list):
                    if len(data) > 0:
                        data = data[0]
                    else:
                        raise ValueError("Model returned empty list")

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

                return {
                    "score": float(weighted_score),
                    "presence": float(data.get("presence", 0)),
                    "layout": float(data.get("layout", 0)),
                    "connectivity": float(data.get("connectivity", 0)),
                    "details": float(data.get("details", 0)),
                    "reasoning": data.get("reasoning", "")
                }

            except (APIConnectionError, RateLimitError, APIError) as e:
                sleep_time = base_delay * (2 ** attempt) + np.random.uniform(0, 1)
                print(f"OpenAI API Error ({type(e).__name__}). Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            
            except Exception as e:
                print(f"GPT Eval Error (Attempt {attempt}): {e}")
                if attempt == max_retries - 1:
                    break
                time.sleep(2)

        return {k: np.nan for k in ["score", "presence", "layout", "connectivity", "details"]}


# ==========================================================
# ====================== Eval Logic =========================
# ==========================================================

def eval_mllm(
    bundle: Bundle,
    gemini_judge: GeminiJudge,
    gpt_judge: GPTJudge,
    existing_records: Optional[List[Dict[str, Any]]] = None,
    force_rerender: bool = False,
    render_size_px: Optional[int] = None,
    flush_every: int = 0,
    render_timeout_sec: int = 30,
    limit: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = list(existing_records) if existing_records else []
    summaries: List[Dict[str, Any]] = []

    # ---- running aggregates (for streaming JSON/MD updates) ----
    metric_cols = [
        "ssim", "lpips", "dino_cos", "clip_cos", "siglip_cos", 
        "gemini_score", "gemini_presence", "gemini_layout", "gemini_connectivity", "gemini_details",
        "gpt_score", "gpt_presence", "gpt_layout", "gpt_connectivity", "gpt_details"
    ]

    @dataclass
    class _MetricAgg:
        s: float = 0.0
        n: int = 0

        def add(self, v: Any) -> None:
            vv = _to_jsonable(v)
            if vv is None:
                return
            fv = float(vv)
            if np.isnan(fv):
                return
            self.s += fv
            self.n += 1

        def mean(self) -> Optional[float]:
            return (self.s / self.n) if self.n > 0 else None

    @dataclass
    class _DatasetAgg:
        dataset: str
        ref_count_total: int
        svg_count_found_total: int
        processed: int = 0
        missing_svg_count: int = 0
        render_success_count: int = 0
        render_fail_count: int = 0
        metrics: Dict[str, _MetricAgg] = None  # type: ignore

        def __post_init__(self) -> None:
            self.metrics = {m: _MetricAgg() for m in metric_cols}

    dset_aggs: Dict[str, _DatasetAgg] = {}

    total_processed = 0
    existing_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for rec in rows:
        dname = str(rec.get("dataset", ""))
        if not dname:
            continue
        existing_by_dataset.setdefault(dname, []).append(rec)

    for dset_name, cfg in DATASETS.items():
        ref_dir: Path = cfg["ref_img_dir"]
        gen_dir = MODEL_SVG_ROOT / cfg["gen_svg_dirname"]

        ref_map = list_ref_images(ref_dir)
        gen_map = list_svgs(gen_dir)

        ref_count = len(ref_map)
        svg_found = len(gen_map)
        svg_found_rate = (svg_found / ref_count) if ref_count > 0 else 0.0

        render_ok_count = 0
        dset_aggs[dset_name] = _DatasetAgg(
            dataset=dset_name,
            ref_count_total=ref_count,
            svg_count_found_total=svg_found,
        )
        agg = dset_aggs[dset_name]

        # Resume support: hydrate aggregates from existing records.
        processed_stems: set[str] = set()
        for rec in existing_by_dataset.get(dset_name, []):
            stem = str(rec.get("img_name", ""))
            if stem:
                processed_stems.add(stem)
            agg.processed += 1
            total_processed += 1
            svg_present = bool(str(rec.get("gen_svg_path", "")).strip())
            render_success = int(rec.get("render_success", 0)) == 1
            if not svg_present:
                agg.missing_svg_count += 1
            elif render_success:
                agg.render_success_count += 1
                render_ok_count += 1
                for m in metric_cols:
                    agg.metrics[m].add(rec.get(m))
            else:
                agg.render_fail_count += 1

        pbar = tqdm(list(ref_map.items()), desc=f"[{MODEL_NAME}] {dset_name}", ncols=110)
        for i, (stem, ref_img_path) in enumerate(pbar):
            if limit > 0 and i >= limit:
                break 
            if stem in processed_stems:
                continue

            ref_img = load_image_rgb(ref_img_path)

            svg_path = gen_map.get(stem, None)
            render_ok = 0

            # Init all metrics to NaN
            ssim_v = np.nan
            lpips_v = np.nan
            dino_v = np.nan
            clip_v = np.nan
            siglip_v = np.nan
            
            gemini_score_v = np.nan
            gemini_presence_v = np.nan
            gemini_layout_v = np.nan
            gemini_conn_v = np.nan
            gemini_details_v = np.nan
            gemini_reasoning_v = ""

            gpt_score_v = np.nan
            gpt_presence_v = np.nan
            gpt_layout_v = np.nan
            gpt_conn_v = np.nan
            gpt_details_v = np.nan
            gpt_reasoning_v = ""

            render_err = None

            if svg_path is not None:
                out_png = RENDER_CACHE_ROOT / MODEL_NAME / dset_name / f"{stem}.png"
                if force_rerender or (not out_png.exists()) or out_png.stat().st_size == 0:
                    render_ok, render_err = render_svg_to_png(
                            svg_path,
                            out_png,
                            size_px=render_size_px,
                            timeout_sec=int(render_timeout_sec),
                        )
                    render_ok = 1 if render_ok else 0
                else:
                    render_ok = 1
                    render_err = None

                if render_ok == 1:
                    render_ok_count += 1
                    try:
                        gen_img = load_image_rgb(out_png)
                        gen_img = resize_to_match(gen_img, ref_img)

                        ssim_v = compute_ssim(ref_img, gen_img)

                        t_ref = torch_image_for_lpips(ref_img, bundle.device)
                        t_gen = torch_image_for_lpips(gen_img, bundle.device)
                        lp = bundle.lpips_model(t_ref, t_gen)
                        lpips_v = float(lp.item())

                        clip_v = cosine(embed_clip(bundle, ref_img), embed_clip(bundle, gen_img))
                        dino_v = cosine(embed_dino(bundle, ref_img), embed_dino(bundle, gen_img))
                        siglip_v = cosine(embed_siglip(bundle, ref_img), embed_siglip(bundle, gen_img))

                        # === Gemini Eval ===
                        gemini_results = gemini_judge.evaluate(ref_img, gen_img)
                        gemini_score_v = gemini_results["score"]
                        gemini_presence_v = gemini_results["presence"]
                        gemini_layout_v = gemini_results["layout"]
                        gemini_conn_v = gemini_results["connectivity"]
                        gemini_details_v = gemini_results["details"]
                        gemini_reasoning_v = gemini_results.get("reasoning", "")

                        # === GPT Eval ===
                        gpt_results = gpt_judge.evaluate(ref_img, gen_img)
                        gpt_score_v = gpt_results["score"]
                        gpt_presence_v = gpt_results["presence"]
                        gpt_layout_v = gpt_results["layout"]
                        gpt_conn_v = gpt_results["connectivity"]
                        gpt_details_v = gpt_results["details"]
                        gpt_reasoning_v = gpt_results.get("reasoning", "")

                    except Exception as e:
                        short_id = stem.split('_')[0]
                        pbar.set_postfix_str(f"FAIL[{type(e).__name__}]:{short_id}")
            else:
                render_err = "missing_svg"

            # ---- update running aggregates ----
            agg = dset_aggs[dset_name]
            agg.processed += 1
            total_processed += 1
            if svg_path is None:
                agg.missing_svg_count += 1
            else:
                if render_ok == 1:
                    agg.render_success_count += 1
                    agg.metrics["ssim"].add(ssim_v)
                    agg.metrics["lpips"].add(lpips_v)
                    agg.metrics["dino_cos"].add(dino_v)
                    agg.metrics["clip_cos"].add(clip_v)
                    agg.metrics["siglip_cos"].add(siglip_v)
                    
                    agg.metrics["gemini_score"].add(gemini_score_v)
                    agg.metrics["gemini_presence"].add(gemini_presence_v)
                    agg.metrics["gemini_layout"].add(gemini_layout_v)
                    agg.metrics["gemini_connectivity"].add(gemini_conn_v)
                    agg.metrics["gemini_details"].add(gemini_details_v)

                    agg.metrics["gpt_score"].add(gpt_score_v)
                    agg.metrics["gpt_presence"].add(gpt_presence_v)
                    agg.metrics["gpt_layout"].add(gpt_layout_v)
                    agg.metrics["gpt_connectivity"].add(gpt_conn_v)
                    agg.metrics["gpt_details"].add(gpt_details_v)
                else:
                    agg.render_fail_count += 1

            rows.append(
                {
                    "model": MODEL_NAME,
                    "dataset": dset_name,
                    "img_name": stem,
                    "ref_img_path": str(ref_img_path),
                    "gen_svg_path": str(svg_path) if svg_path is not None else "",
                    "render_success": int(render_ok),
                    "render_error": render_err,
                    "ssim": ssim_v,
                    "lpips": lpips_v,
                    "dino_cos": dino_v,
                    "clip_cos": clip_v,
                    "siglip_cos": siglip_v,
                    "gemini_score": gemini_score_v,
                    "gemini_presence": gemini_presence_v,
                    "gemini_layout": gemini_layout_v,
                    "gemini_connectivity": gemini_conn_v,
                    "gemini_details": gemini_details_v,
                    "gemini_reasoning": gemini_reasoning_v,
                    "gpt_score": gpt_score_v,
                    "gpt_presence": gpt_presence_v,
                    "gpt_layout": gpt_layout_v,
                    "gpt_connectivity": gpt_conn_v,
                    "gpt_details": gpt_details_v,
                    "gpt_reasoning": gpt_reasoning_v
                }
            )
            processed_stems.add(stem)

            if flush_every and (total_processed % int(flush_every) == 0):
                try:
                    _write_streaming_outputs(rows=rows, dset_aggs=dset_aggs, stream_records=False)
                except Exception:
                    pass

        render_success_rate = (render_ok_count / ref_count) if ref_count > 0 else 0.0

        df_dset = pd.DataFrame([r for r in rows if r["dataset"] == dset_name])
        df_ok = df_dset[df_dset["render_success"] == 1]

        summaries.append(
            {
                "model": MODEL_NAME,
                "dataset": dset_name,
                "ref_count": ref_count,
                "svg_count_found": svg_found,
                "svg_found_rate": svg_found_rate,
                "render_success_count": int(render_ok_count),
                "render_success_rate": render_success_rate,
                "avg_ssim": float(df_ok["ssim"].mean()) if len(df_ok) else np.nan,
                "avg_lpips": float(df_ok["lpips"].mean()) if len(df_ok) else np.nan,
                "avg_dino_cos": float(df_ok["dino_cos"].mean()) if len(df_ok) else np.nan,
                "avg_clip_cos": float(df_ok["clip_cos"].mean()) if len(df_ok) else np.nan,
                "avg_siglip_cos": float(df_ok["siglip_cos"].mean()) if len(df_ok) else np.nan,
                "avg_gemini_score": float(df_ok["gemini_score"].mean()) if len(df_ok) else np.nan,
                "avg_gpt_score": float(df_ok["gpt_score"].mean()) if len(df_ok) else np.nan,
            }
        )

        if flush_every:
            try:
                _write_streaming_outputs(rows=rows, dset_aggs=dset_aggs, stream_records=False)
            except Exception:
                pass

    return pd.DataFrame(rows), pd.DataFrame(summaries)


def load_existing_records(per_image_json_path: Path, expected_model: str) -> List[Dict[str, Any]]:
    if not per_image_json_path.exists():
        return []
    try:
        payload = json.loads(per_image_json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    if str(payload.get("model", "")) != expected_model:
        return []
    records = payload.get("records", [])
    if not isinstance(records, list):
        return []
    out: List[Dict[str, Any]] = []
    for rec in records:
        if isinstance(rec, dict) and rec.get("dataset") and rec.get("img_name"):
            out.append(rec)
    return out

def _to_jsonable(x: Any) -> Any:
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if np.isnan(v) else v
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, torch.Tensor) and x.ndim == 0:
        return _to_jsonable(x.item())
    return x

def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def _build_running_stats(dset_aggs: Dict[str, Any]) -> Dict[str, Any]:
    metric_cols = [
        "ssim", "lpips", "dino_cos", "clip_cos", "siglip_cos",
        "gemini_score", "gemini_presence", "gemini_layout", "gemini_connectivity", "gemini_details",
        "gpt_score", "gpt_presence", "gpt_layout", "gpt_connectivity", "gpt_details"
    ]

    datasets: Dict[str, Any] = {}
    total_images = 0
    total_svg_found = 0
    total_missing_svg = 0
    total_render_success = 0
    total_render_fail = 0
    overall_aggs = {m: {"s": 0.0, "n": 0} for m in metric_cols}

    for name, agg in dset_aggs.items():
        total_images += int(agg.processed)
        total_svg_found += int(min(agg.svg_count_found_total, agg.ref_count_total))
        total_missing_svg += int(agg.missing_svg_count)
        total_render_success += int(agg.render_success_count)
        total_render_fail += int(agg.render_fail_count)

        per_means = {m: _to_jsonable(agg.metrics[m].mean()) for m in metric_cols}
        datasets[str(name)] = {
            "ref_count_total": int(agg.ref_count_total),
            "svg_count_found_total": int(agg.svg_count_found_total),
            "processed": int(agg.processed),
            "missing_svg_count": int(agg.missing_svg_count),
            "render_success_count": int(agg.render_success_count),
            "render_fail_count": int(agg.render_fail_count),
            "avg_metrics_on_render_success_so_far": per_means,
        }

        for m in metric_cols:
            overall_aggs[m]["s"] += float(agg.metrics[m].s)
            overall_aggs[m]["n"] += int(agg.metrics[m].n)

    overall_means = {
        m: (_to_jsonable(overall_aggs[m]["s"] / overall_aggs[m]["n"]) if overall_aggs[m]["n"] > 0 else None)
        for m in metric_cols
    }

    return {
        "total_processed": int(total_images),
        "total_missing_svg": int(total_missing_svg),
        "total_render_success": int(total_render_success),
        "total_render_fail": int(total_render_fail),
        "avg_metrics_on_render_success_so_far": overall_means,
        "datasets": datasets,
    }

def _write_streaming_outputs(rows: List[Dict[str, Any]], dset_aggs: Dict[str, Any], stream_records: bool = False) -> None:
    stats = _build_running_stats(dset_aggs)
    per_image_json = EVAL_OUT_ROOT / f"{MODEL_NAME}__per_image.json"
    summary_md = EVAL_OUT_ROOT / f"{MODEL_NAME}__summary.md"

    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "benchmark_root": str(BENCHMARK_ROOT),
        "model_svg_root": str(MODEL_SVG_ROOT),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "running": stats,
    }
    if stream_records:
        payload["records"] = [{k: _to_jsonable(v) for k, v in rec.items()} for rec in rows]

    _atomic_write_text(per_image_json, json.dumps(payload, indent=2, ensure_ascii=False))

    lines: List[str] = []
    lines.append(f"# {MODEL_NAME} evaluation summary (running)")
    lines.append("")
    lines.append(f"- Updated at: `{payload['generated_at']}`")
    lines.append(f"- Processed: **{stats['total_processed']}** images")
    lines.append(f"- Render success: **{stats['total_render_success']}** | Render fail: **{stats['total_render_fail']}** | Missing SVG: **{stats['total_missing_svg']}**")
    lines.append("")
    lines.append("## Per-dataset (running)")
    lines.append("")
    cols = [
        "dataset", 
        "processed", 
        "render_success", 
        "avg_ssim", 
        "avg_lpips", 
        "avg_dino", 
        "avg_clip", 
        "avg_siglip",
        "avg_gemini",
        "avg_gpt"
    ]
    lines.append("| " + " | ".join(cols) + " |")
    align = [":---"] + ["---:"] * (len(cols)-1)
    lines.append("| " + " | ".join(align) + " |")
    for dset, d in sorted(stats["datasets"].items(), key=lambda x: x[0]):
        m = d["avg_metrics_on_render_success_so_far"]
        row = [
            dset,
            str(d["processed"]),
            str(d["render_success_count"]),
            "" if m.get("ssim") is None else f"{float(m['ssim']):.4f}",
            "" if m.get("lpips") is None else f"{float(m['lpips']):.4f}",
            "" if m.get("dino_cos") is None else f"{float(m['dino_cos']):.4f}",
            "" if m.get("clip_cos") is None else f"{float(m['clip_cos']):.4f}",
            "" if m.get("siglip_cos") is None else f"{float(m['siglip_cos']):.4f}",
            "" if m.get("gemini_score") is None else f"{float(m['gemini_score']):.4f}",
            "" if m.get("gpt_score") is None else f"{float(m['gpt_score']):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    _atomic_write_text(summary_md, "\n".join(lines) + "\n")

def write_summary_md(summary_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "dataset",
        "ref_count",
        "svg_count_found",
        "render_success_count",
        "avg_ssim",
        "avg_lpips",
        "avg_dino_cos",
        "avg_clip_cos",
        "avg_siglip_cos",
        "avg_gemini_score",
        "avg_gpt_score"
    ]

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        try:
            fv = float(v)
            if np.isnan(fv):
                return ""
            return f"{fv:.4f}"
        except Exception:
            return str(v)

    lines = []
    lines.append(f"# {MODEL_NAME} evaluation summary")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Benchmark root: `{BENCHMARK_ROOT}`")
    lines.append(f"- Model SVG root: `{MODEL_SVG_ROOT}`")
    lines.append("")
    lines.append("## Per-dataset summary")
    lines.append("")

    header = "| " + " | ".join(cols) + " |"
    align = [":---"] + ["---:"] * (len(cols)-1)
    sep = "| " + " | ".join(align) + " |"
    lines.append(header)
    lines.append(sep)

    for _, row in summary_df.iterrows():
        r = {k: _to_jsonable(row.get(k, None)) for k in cols}
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")

    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_per_image_json(per_image_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = per_image_df.copy()

    for c in ["render_success"]:
        if c in df.columns:
            df[c] = df[c].astype(int)

    datasets_stats: Dict[str, Any] = {}
    overall_ok = df[df["render_success"] == 1]

    metric_cols = [
        "ssim", "lpips", "dino_cos", "clip_cos", "siglip_cos",
        "gemini_score", "gemini_presence", "gemini_layout", "gemini_connectivity", "gemini_details",
        "gpt_score", "gpt_presence", "gpt_layout", "gpt_connectivity", "gpt_details"
    ]

    for dset, df_dset in df.groupby("dataset", sort=True):
        ref_count = int(len(df_dset))
        svg_present = df_dset["gen_svg_path"].astype(str) != ""
        svg_count_found = int(svg_present.sum())
        missing_svg_count = int((~svg_present).sum())
        render_success_count = int(df_dset["render_success"].sum())
        render_fail_count = int(((df_dset["render_success"] == 0) & svg_present).sum())

        df_ok = df_dset[df_dset["render_success"] == 1]
        means = {m: _to_jsonable(df_ok[m].mean()) if m in df_ok.columns else None for m in metric_cols}

        datasets_stats[str(dset)] = {
            "ref_count": ref_count,
            "svg_count_found": svg_count_found,
            "svg_found_rate": (svg_count_found / ref_count) if ref_count > 0 else 0.0,
            "missing_svg_count": missing_svg_count,
            "render_success_count": render_success_count,
            "render_success_rate": (render_success_count / ref_count) if ref_count > 0 else 0.0,
            "render_fail_count": render_fail_count,
            "avg_metrics_on_render_success": means,
        }

    overall = {
        "total_images": int(len(df)),
        "total_svg_found": int((df["gen_svg_path"].astype(str) != "").sum()),
        "total_render_success": int((df["render_success"] == 1).sum()),
        "total_render_fail": int((df["render_success"] == 0).sum()),
        "avg_metrics_on_render_success": {m: _to_jsonable(overall_ok[m].mean()) if m in overall_ok.columns else None for m in metric_cols},
    }

    payload = {
        "model": MODEL_NAME,
        "benchmark_root": str(BENCHMARK_ROOT),
        "model_svg_root": str(MODEL_SVG_ROOT),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": {k: _to_jsonable(v) for k, v in overall.items()},
        "datasets": datasets_stats,
        "records": [{k: _to_jsonable(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")],
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

# ==========================================================
# ====================== Main ===============================
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda / cuda:0 / cpu")
    parser.add_argument("--force-rerender", action="store_true", help="re-render all SVG->PNG even if cached")
    parser.add_argument("--render-size", type=int, default=0, help="optional render size (px). 0 = default")
    parser.add_argument("--flush-every", type=int, default=50, help="flush JSON/MD every N images (0 disables)")
    parser.add_argument("--render-timeout", type=int, default=30, help="SVG->PNG render timeout per image (seconds)")
    parser.add_argument("--limit", type=int, default=0, help="Quick run: limit images per dataset (0=all)")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume by skipping images already present in existing per_image.json records (default: on).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and recompute from scratch.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--stream-records",
        action="store_true",
        help="Also include full per-image records in the periodically updated JSON (slower).",
    )
    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    render_size_px = None if args.render_size <= 0 else int(args.render_size)

    EVAL_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RENDER_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"Evaluating model: {MODEL_NAME}")
    print(f"Benchmark root: {BENCHMARK_ROOT}")
    print(f"Model SVG root: {MODEL_SVG_ROOT}")
    print(f"Output dir:     {EVAL_OUT_ROOT}")
    print("=" * 90)

    print("Loading metric backbones (LPIPS + CLIP + DINO + SigLIP)...")
    bundle = load_bundle(device)

    print("Initializing LLM Judges (Gemini & GPT)...")
    gemini_judge = GeminiJudge(model_id="gemini-3-flash-preview") 
    gpt_judge = GPTJudge(model_id="gpt-5.2")

    per_image_json = EVAL_OUT_ROOT / f"{MODEL_NAME}__per_image.json"
    existing_records: List[Dict[str, Any]] = []
    if args.resume:
        existing_records = load_existing_records(per_image_json, MODEL_NAME)
        print(f"Resume mode: loaded {len(existing_records)} existing records from {per_image_json}")

    per_image_df, summary_df = eval_mllm(
        bundle=bundle,
        gemini_judge=gemini_judge,
        gpt_judge=gpt_judge,
        existing_records=existing_records,
        force_rerender=args.force_rerender,
        render_size_px=render_size_px,
        flush_every=int(args.flush_every),
        render_timeout_sec=int(args.render_timeout),
        limit=args.limit,
    )

    summary_md = EVAL_OUT_ROOT / f"{MODEL_NAME}__summary.md"

    write_per_image_json(per_image_df, per_image_json)
    write_summary_md(summary_df, summary_md)

    print(f"\nSaved per-image JSON: {per_image_json}")
    print(f"Saved summary MD:     {summary_md}")
    print("Done.")

if __name__ == "__main__":
    main()