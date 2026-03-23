import argparse
import os
import json
import random
from PIL import Image
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# --- CONFIGURATION ---
# Set IMAGE_ROOT to the root directory containing your training images.
# Set SOURCE_FILES to point to your annotation JSON files.
IMAGE_ROOT = "/path/to/your/image/root"

SOURCE_FILES = [
    {
        "path": "/path/to/finetune_data_starvector.json",
        "source_name": "starvector"
    },
    {
        "path": "/path/to/finetune_data_molmo.json",
        "source_name": "molmo"
    },
    {
        "path": "/path/to/finetune_data_synthetic_svg.json",
        "source_name": "synthetic"
    },
    {
        "path": "/path/to/finetune_data_arxiv_p2f.json",
        "source_name": "arxiv_p2f"
    },
    {
        "path": "/path/to/finetune_data_arxiv.json",
        "source_name": "arxiv"
    }
]


def process_item(item, source_name):
    """Parses a single JSON item, preserving the exact SFT prompts."""

    # 1. Resolve Image Path
    try:
        rel_path = item["images"][0]
        full_img_path = os.path.join(IMAGE_ROOT, rel_path)
    except (IndexError, KeyError, TypeError):
        return None

    # 2. Get Dimensions
    if not os.path.exists(full_img_path):
        return None
    try:
        with Image.open(full_img_path) as im:
            W, H = im.size
    except Exception:
        return None

    # 3. Extract User Prompt AND Assistant Response (Ground Truth)
    user_prompt = ""
    ground_truth_text = ""

    for msg in item.get("messages", []):
        if msg["role"] == "user":
            user_prompt = msg["content"]
        elif msg["role"] == "assistant":
            ground_truth_text = msg["content"]

    if not user_prompt or not ground_truth_text:
        return None

    # 4. Build RL Sample
    return {
        "data_source": source_name,
        "prompt": [
            {"role": "user", "content": user_prompt},
        ],
        "response": ground_truth_text,
        "ability": "svg_gen",
        "reward_model": {
            "style": "rule",
            "ground_truth": full_img_path,
            "ground_truth_code": ground_truth_text
        },
        "extra_info": {
            "png_size": [int(W), int(H)],
        },
        "images": [full_img_path],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/combined_axiv_molmo_star")
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--val_size", type=int, default=-1)
    args = parser.parse_args()

    all_data = []

    # 1. Load and Process All Files
    for entry in SOURCE_FILES:
        path = entry["path"]
        src = entry["source_name"]

        if not os.path.exists(path):
            print(f"Skipping missing file: {path}", flush=True)
            continue

        print(f"Loading {src} from {path}...", flush=True)
        with open(path, 'r') as f:
            raw_data = json.load(f)

        print(f"  -> Found {len(raw_data)} items. Processing...", flush=True)

        count = 0
        for item in tqdm(raw_data, desc=f"Processing {src}"):
            processed = process_item(item, src)
            if processed:
                all_data.append(processed)
                count += 1
        print(f"  -> Successfully processed {count} items.\n", flush=True)

    print(f"Total combined items: {len(all_data)}", flush=True)

    # 2. Shuffle
    print("Shuffling data...", flush=True)
    random.seed(42)
    random.shuffle(all_data)

    # 3. Split Train/Test (90/10)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # 4. Apply Limits
    if args.train_size > 0:
        train_data = train_data[:args.train_size]
    if args.val_size > 0:
        val_data = val_data[:args.val_size]

    print(f"Final Counts -> Train: {len(train_data)} | Test: {len(val_data)}", flush=True)

    # 5. Save
    print(f"Saving to {args.output_dir}...", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    ds_train = Dataset.from_list(train_data)
    ds_test = Dataset.from_list(val_data)

    ds_train.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    ds_test.to_parquet(os.path.join(args.output_dir, "test.parquet"))

    print(f"Done! Saved parquet files to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
