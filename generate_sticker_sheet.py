import os
import re
import cv2
import time
import torch
import hashlib
import subprocess
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from fuzzywuzzy import fuzz
from transformers import BlipProcessor, BlipForConditionalGeneration

# Setup DeepSeek via OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "deepseek/deepseek-r1:free"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def remove_background_near_color(img, target_color=(240, 230, 210), threshold=30):
    data = np.array(img)
    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    mask = ((np.abs(r - target_color[0]) < threshold) &
            (np.abs(g - target_color[1]) < threshold) &
            (np.abs(b - target_color[2]) < threshold))
    data[..., 3][mask] = 0
    return Image.fromarray(data)

def hash_image(image: Image.Image):
    return hashlib.md5(image.tobytes()).hexdigest()

def describe_image(image: Image.Image):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True).strip().lower()

def is_caption_relevant(caption, keywords, threshold=65):
    return any(fuzz.partial_ratio(kw, caption) >= threshold for kw in keywords)

def create_kisscut_sticker(cropped_image):
    rgba = np.array(cropped_image)
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cropped_image
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    kernel = np.ones((10, 10), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    outline_rgba = np.zeros_like(rgba)
    outline_rgba[dilated == 255] = [255, 255, 255, 255]
    result = Image.fromarray(outline_rgba)
    result.paste(cropped_image, (0, 0), cropped_image)
    bbox = result.getbbox()
    return result.crop(bbox)

def place_on_canvas(canvas, stickers, margin=100, padding=60):
    cell_w, cell_h = 500 + padding, 500 + padding
    col, row = 0, 0
    for sticker in stickers:
        s = sticker.copy()
        s.thumbnail((500, 500), Image.Resampling.LANCZOS)
        x = margin + col * cell_w
        y = margin + row * cell_h
        canvas.paste(s, (x, y), s)
        col += 1
        if x + 2 * cell_w > canvas.width:
            col = 0
            row += 1

def get_latest_drop_folder(drops_dir="drops"):
    folders = [os.path.join(drops_dir, f) for f in os.listdir(drops_dir)
               if os.path.isdir(os.path.join(drops_dir, f))]
    return max(folders, key=os.path.getmtime)

def load_keywords(prompt_path, keywords_path):
    keywords = []
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().lower()
            keywords += re.findall(r'\b\w+\b', prompt_text)
    if os.path.exists(keywords_path):
        with open(keywords_path, "r", encoding="utf-8") as f:
            keywords += [line.strip().lower() for line in f if line.strip()]
    return list(set(keywords))

def deepseek_validate_captions(theme, captions):
    joined = "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(captions)])
    messages = [
        { "role": "system", "content": "You are a creative art director. You are strict but fair." },
        { "role": "user", "content": f"""The following captions are descriptions of stickers. The theme is "{theme}". 
Please return a comma-separated list of numbers of any captions that clearly do NOT match the theme.

{joined}
"""}
    ]
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": 100,
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
    content = response.json()["choices"][0]["message"]["content"]
    return [int(s.strip()) for s in content.split(",") if s.strip().isdigit()]

def generate_and_fill_stickersheet():
    REQUIRED = 12
    MAX_ATTEMPTS = 20
    valid_stickers = []
    valid_captions = []
    seen_hashes = set()
    os.makedirs("extracted_stickers", exist_ok=True)

    for attempt in range(MAX_ATTEMPTS):
        print(f"\n🔁 Attempt {attempt + 1}")
        subprocess.run(["python", "generate_images.py"])
        time.sleep(1)

        drop = get_latest_drop_folder()
        sheet = os.path.join(drop, "stickersheet.png")
        prompt = os.path.join(drop, "prompt.txt")
        keywords_file = os.path.join(drop, "keywords.txt")

        if not os.path.exists(sheet) or not os.path.exists(prompt):
            print("❌ Missing required files.")
            continue

        keywords = load_keywords(prompt, keywords_file)
        theme = os.path.basename(drop).split("_")[0]

        original = Image.open(sheet).convert("RGBA")
        original = remove_background_near_color(original)
        white_bg = Image.new("RGBA", original.size, "WHITE")
        flattened = Image.alpha_composite(white_bg, original)

        cv_img = cv2.cvtColor(np.array(flattened), cv2.COLOR_RGBA2BGRA)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2GRAY)

        # ✅ Better thresholding for beige/light backgrounds
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Optional: dilate slightly to ensure clean outlines
        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
        print(f"🧩 Found {len(contours)} potential sticker shapes")

        # Debug visual: save outline of all stickers found
        debug_img = cv_img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0, 255), 2)
        cv2.imwrite(os.path.join(drop, "contour_debug.png"), debug_img)

        for cnt in contours:
            if len(valid_stickers) >= REQUIRED:
                break

            area = cv2.contourArea(cnt)
            if area < 3000:
                continue  # ✅ Remove max size limit, only skip small dots

            x, y, w, h = cv2.boundingRect(cnt)

            mask = np.zeros(original.size[::-1], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            extracted = Image.new("RGBA", original.size, (0, 0, 0, 0))
            extracted.paste(original, mask=Image.fromarray(mask))
            cropped = extracted.crop((x, y, x + w, y + h))
            with_outline = create_kisscut_sticker(cropped)

            if not with_outline.getbbox():
                continue

            img_hash = hash_image(with_outline)
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            caption = describe_image(with_outline)
            if not is_caption_relevant(caption, keywords):
                print(f"⚠️ Rejected: '{caption}'")
                continue

            idx = len(valid_stickers) + 1
            out_path = os.path.join("extracted_stickers", f"sticker_{idx:02}.png")
            with_outline.save(out_path)
            valid_stickers.append(with_outline)
            valid_captions.append(caption)
            print(f"✅ Accepted: {caption} ({idx}/{REQUIRED})")

        if len(valid_stickers) >= REQUIRED:
            break

    if len(valid_stickers) < REQUIRED:
        print("❌ Failed to gather 12 valid stickers.")
        return

    # Optionally validate again with DeepSeek after this point (if needed)

    canvas = Image.new("RGBA", (2480, 3508), "WHITE")
    place_on_canvas(canvas, valid_stickers)
    final_output = os.path.join(drop, "a4_stickersheet.png")
    canvas.save(final_output)
    print(f"\n✅ Final A4 sticker sheet saved: {final_output}")


if __name__ == "__main__":
    generate_and_fill_stickersheet()
