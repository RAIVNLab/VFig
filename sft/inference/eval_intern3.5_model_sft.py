import os
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# -------------------------------------------------------------
# ARGS (sharding)
# -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_shards", type=int, default=1, help="Total number of parallel jobs")
parser.add_argument("--shard_id", type=int, default=0, help="This job's shard id [0..num_shards-1]")
args = parser.parse_args()

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
base_model = "OpenGVLab/InternVL3_5-4B-HF"
adapter_path = "xxx"

benchmark_root = "xxx"
input_root = os.path.join(benchmark_root, "original_img")
output_root = os.path.join(benchmark_root, "rendered_svg", "xxx")

# input_folder_name -> output_folder_name
DATASETS = {
    "arxiv_p2f_img_eval_397": "arxiv_p2f_rendered_svg_397",
    "molmo_img_eval_499": "molmo_rendered_svg_499",
    "starvector_img_eval_440": "starvector_rendered_svg_440",
}

valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Prompt (minimal, benchmark-friendly)
PROMPT_TEXT = "Convert this figure into valid SVG code."

# Generation
MAX_NEW_TOKENS = 8192

# -------------------------------------------------------------
# LOAD MODEL + PROCESSOR (ONCE PER JOB / GPU)
# -------------------------------------------------------------
print(f"[Shard {args.shard_id}/{args.num_shards}] Loading processor...")
processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

print(f"[Shard {args.shard_id}/{args.num_shards}] Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"[Shard {args.shard_id}/{args.num_shards}] Loading adapter...")
model.load_adapter(adapter_path)
model.eval()

# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def extract_svg(text: str) -> str:
    """Keep only the <svg ... </svg> block if present."""
    if "<svg" in text:
        text = text[text.find("<svg"):]
    if "</svg>" in text:
        text = text[: text.find("</svg>") + len("</svg>")]
    return text.strip()

def img_to_svg(img: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    chat_input = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = processor(
        text=[chat_input],
        images=[img],
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    decoded = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_svg(decoded)

# -------------------------------------------------------------
# PROCESS DATASETS (SHARDED)
# -------------------------------------------------------------
os.makedirs(output_root, exist_ok=True)

total_done = 0
total_skip = 0
total_fail = 0

print(f"[Shard {args.shard_id}/{args.num_shards}] Starting datasets...")

for in_folder_name, out_folder_name in DATASETS.items():
    in_dir = os.path.join(input_root, in_folder_name)
    out_dir = os.path.join(output_root, out_folder_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(in_dir):
        print(f"[WARN] Missing input dir: {in_dir} (skip)")
        continue

    # List ALL files, then shard by args.shard_id / args.num_shards
    all_files = sorted(os.listdir(in_dir))
    files = all_files[args.shard_id :: args.num_shards]

    print(
        f"\n=== Dataset: {in_folder_name} -> {out_folder_name} | "
        f"total={len(all_files)} | mine={len(files)} | shard={args.shard_id}/{args.num_shards} ==="
    )

    for fname in files:
        name, ext = os.path.splitext(fname)
        if ext.lower() not in valid_ext:
            continue

        img_path = os.path.join(in_dir, fname)
        svg_path = os.path.join(out_dir, f"{name}_adapter.svg")

        # resume-friendly (NO reprocess if already good)
        if os.path.exists(svg_path):
            sz = os.path.getsize(svg_path)
            if sz > 50:
                total_skip += 1
                print(f"[SKIP] {fname} -> {os.path.basename(svg_path)} (exists, {sz} bytes)")
                continue
            else:
                print(f"[WARN] Tiny existing file (will regenerate): {svg_path} ({sz} bytes)")

        try:
            print(f"Processing: {img_path}")
            img = Image.open(img_path).convert("RGB")

            svg_text = img_to_svg(img)

            # basic sanity: ensure we at least have an <svg
            if "<svg" not in svg_text:
                raise ValueError("Model output did not contain <svg>")

            with open(svg_path, "w") as f:
                f.write(svg_text)

            total_done += 1
            print(f"Saved SVG -> {svg_path}")

        except Exception as e:
            total_fail += 1
            print(f"[FAIL] {img_path} | {type(e).__name__}: {e}")

print("\nDone.")
print(f"[Shard {args.shard_id}/{args.num_shards}] Converted: {total_done} | Skipped(existing): {total_skip} | Failed: {total_fail}")
print(f"Output root: {output_root}")
