import argparse
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_NAME = "XunmeiLiu/VFIG-4B"
DEFAULT_IMAGE = "images/example1.png"
PROMPT = "Generate the SVG code for this image."


def load_model(model_name: str):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def generate_svg(image_path: str, model, processor, max_new_tokens: int = 4096) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate SVG from an image using VFig.")
    parser.add_argument("image", type=str, nargs="?", default=DEFAULT_IMAGE, help="Path to input image (default: images/example1.png)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output SVG (optional)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="HuggingFace model name or local path")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, processor = load_model(args.model)

    print(f"Generating SVG for: {args.image}")
    svg = generate_svg(args.image, model, processor, args.max_new_tokens)

    if args.output:
        with open(args.output, "w") as f:
            f.write(svg)
        print(f"SVG saved to: {args.output}")
    else:
        print(svg)


if __name__ == "__main__":
    main()
