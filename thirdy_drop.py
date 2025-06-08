import os
import json
import uuid
import requests
import re
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "deepseek/deepseek-r1:free"
DROP_DIR = "drops"

def extract_allowed_adjectives(theme):
    return [w.lower() for w in re.findall(r"\b\w+\b", theme) if w.isalpha()]

def expand_prompt(theme: str) -> str:
    allowed_adjectives = extract_allowed_adjectives(theme)
    adjective_clause = ", ".join(allowed_adjectives)

    system_instruction = (
        "You are a visual prompt generator for a Stable Diffusion XL sticker sheet LoRA. "
        "Generate a **single** highly optimized, clean visual prompt. "
        "You must follow the following rules strictly:\n"
        "1. The final prompt MUST contain the word 'StickerSheet'.\n"
        "2. ONLY use adjectives or descriptive keywords found in the user theme.\n"
        "3. Keep the structure: '[Theme], StickerSheet, [optional allowed adjectives], high quality, sharp image.'\n"
        "4. NEVER include the words 'photo', 'photorealistic', 'render', or modifiers unrelated to the input.\n"
        "Only output the prompt — no explanation, no markdown."
    )

    prompt = f"""
Theme: '{theme}'
Allowed words: {adjective_clause}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "X-Title": "thirdy-single-prompt",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if not response.ok:
        raise Exception(f"❌ Prompt API error {response.status_code}: {response.text}")
    raw = response.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    if not raw:
        raise ValueError("❌ Empty prompt.")
    return raw

def generate_keywords(theme: str) -> list:
    system_instruction = (
        "You are a keyword generator for visual and conceptual themes. "
        "Given a creative theme, return up to 30 single, distinct English words that are clearly and visually associated with it. "
        "Do NOT include the original theme phrase or any part of it (e.g., if the theme is 'royal frogs', don't include 'royal' or 'frogs'). "
        "Words must be lowercase, only one word each, no punctuation or numbers, and separated by commas. "
        "Focus on nouns and adjectives that describe objects, animals, symbols, colors, or concepts related to the theme."
    )

    user_input = f"Generate 30 associated single words for the theme: {theme}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "X-Title": "thirdy-associated-keywords",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input},
        ],
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if not response.ok:
        raise Exception(f"❌ Keywords API error {response.status_code}: {response.text}")
    raw = response.json()["choices"][0]["message"]["content"].strip()

    # Clean and split
    keywords = [kw.strip().lower() for kw in re.split(r"[,\n]", raw) if kw.strip()]
    return [kw for kw in keywords if " " not in kw][:30]


def run_drop(theme: str):
    print(f"\n🎨 Generating assets for theme: {theme}")
    prompt = expand_prompt(theme)
    keywords = generate_keywords(theme)

    drop_id = str(uuid.uuid4())[:8]
    folder_name = f"{theme.replace(' ', '_').lower()}_{drop_id}"
    drop_path = os.path.join(DROP_DIR, folder_name)
    os.makedirs(drop_path, exist_ok=True)

    prompt_path = os.path.join(drop_path, "prompt.txt")
    keywords_path = os.path.join(drop_path, "keywords.txt")

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(keywords_path, "w", encoding="utf-8") as f:
        f.write("\n".join(keywords))

    print(f"\n✅ Saved prompt to: {prompt_path}")
    print(f"✅ Saved keywords to: {keywords_path}")
    print(f"🧠 Prompt: {prompt}")
    print(f"🔑 Keywords: {', '.join(keywords)}")

if __name__ == "__main__":
    theme = input("Enter theme for this drop (e.g. Royal Frogs): ").strip()
    run_drop(theme)
