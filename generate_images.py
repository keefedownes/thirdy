import os
import json
import requests
import base64
from pathlib import Path
from time import sleep

# === Config ===
A1111_API_URL = "http://127.0.0.1:7860"
LORA_NAME = "stickersheet"  # Must match filename of your LoRA in /models/Lora
WIDTH = 1024
HEIGHT = 1024
NEGATIVE_PROMPT = "text, blurry, deformed, cropped, lowres, nsfw, extra limbs, watermark"
DROPS_ROOT = Path("drops")
DELAY_BETWEEN_IMAGES = 1  # seconds
OUTPUT_IMAGE_COUNT = 1  # Only one prompt = one image

# === Functions ===

def get_latest_drop_folder():
    all_folders = [f for f in DROPS_ROOT.iterdir() if f.is_dir()]
    if not all_folders:
        raise FileNotFoundError("❌ No drop folders found in /drops/")
    return max(all_folders, key=lambda d: d.stat().st_mtime)

def generate_image(prompt, output_path):
    full_prompt = f"<lora:{LORA_NAME}:1> {prompt}"
    payload = {
        "prompt": full_prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": 20,
        "width": WIDTH,
        "height": HEIGHT,
        "sampler_index": "DPM++ 2M Karras",
        "cfg_scale": 8,
        "seed": -1,
        "batch_size": 1,
        "n_iter": 1,
    }

    try:
        res = requests.post(f"{A1111_API_URL}/sdapi/v1/txt2img", json=payload)
        if not res.ok:
            print(f"❌ HTTP {res.status_code} error: {res.text}")
            return None

        data = res.json()
        if "images" not in data:
            print(f"❌ Response missing 'images':\n{json.dumps(data, indent=2)}")
            return None

        image_data = data["images"][0]
        image_bytes = base64.b64decode(image_data)
        filename = "stickersheet.png"
        out_path = output_path / filename
        with open(out_path, "wb") as f:
            f.write(image_bytes)

        return filename

    except Exception as e:
        print(f"❌ Exception during image generation: {e}")
        return None

def main():
    drop_path = get_latest_drop_folder()
    prompt_file = drop_path / "prompt.txt"
    metadata_file = drop_path / "metadata.json"
    output_path = drop_path

    if not prompt_file.exists():
        raise FileNotFoundError(f"❌ Missing {prompt_file.name} in {drop_path}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    print(f"\n🎨 Generating image for: {drop_path.name}")
    print(f"🧠 Prompt: {prompt}")

    filename = generate_image(prompt, output_path)
    metadata = {
        "theme": drop_path.stem,
        "prompt": prompt,
        "filename": filename or None,
    }

    if not filename:
        metadata["error"] = "Generation failed"

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Done! Image + metadata saved in: {drop_path}")

if __name__ == "__main__":
    main()
