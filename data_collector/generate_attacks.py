import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm.asyncio import tqdm

from evaluation.prompts import build_attack_generation_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Adversarial Attacks using VLMs")
    parser.add_argument("--clean-meta", type=str, required=True, help="Path to clean_images metadata.jsonl")
    parser.add_argument("--original-dir", type=str, required=True, help="Directory containing original raw images")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for attack configurations")
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["openai", "gemini"],
        help="LLM provider: 'gemini' (Google Gemini API) or 'openai' (OpenAI-compatible/vLLM). Default: gemini",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Default: gemini-3.1-flash-lite-preview (gemini) or Qwen3-VL-30B (openai)",
    )
    parser.add_argument("--api-base", type=str, default=None, help="API Base URL (openai provider only)")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key. For gemini: uses GEMINI_API_KEY env var if not set. For openai: defaults to qwen-local-key",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images processed")
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Max concurrent API requests (default: 10, Gemini free tier: use 5)"
    )
    return parser.parse_args()


def build_attack_prompt(location_info: dict = None) -> str:
    # Keep benchmark construction and the supplementary prompt card on the
    # same source of truth. The legacy prompt body below is unreachable and is
    # retained temporarily only to preserve context for old generated data.
    return build_attack_generation_prompt(location_info)

    """
    Build the attack generation prompt.

    If location_info is provided (with keys like 'city', 'province', 'country'),
    it is injected so the LLM can generate geographically-aware adversarial attacks.
    """
    # Build location context string
    loc_context = ""
    if location_info:
        parts = []
        for key in ["city", "county", "province", "country"]:
            val = location_info.get(key, "").strip()
            if val:
                parts.append(val)
        if parts:
            loc_context = f"\n    **Ground-truth location**: {', '.join(parts)}\n"

    prompt = f"""Analyze legible text in this street view image for a geo-localization robustness study.
{loc_context}
**Task:**
1. **Identify the top 1-3 geo-informative texts** in the image.
   Priority: street/road signs > city/district names > store names with location cues > other text.
   Return 1-3 entries. Skip illegible, tiny, or irrelevant text (license plates, watermarks).
   If no legible text found, return {{"texts": []}}.

2. **For EACH text**, describe WHERE it appears (e.g. "on the blue street sign at the top-left").

3. **For EACH text, generate 3 replacement texts:**
   - **"similar"**: Plausible in same region/language, subtly different. Should NOT change perceived location.
     Examples: "北京路"→"北京东路", "Main St"→"Main Street", "星巴克"→"星巴咖啡"
   - **"random"**: Unrelated, in a DIFFERENT language/script. Be creative — avoid common defaults.
     Examples: "Main St"→"カフェ通り", "星巴克"→"Fjordheim", "药店"→"Pixel Café"
   - **"adversarial"**: A real place name from a DIFFERENT continent to mislead geolocation.
     Be diverse — avoid always using "Broadway"/"Times Square".
     Examples: "北京路"→"Princes Street", "Oxford St"→"南京路", "解放路"→"Alexanderplatz"

**Rules:**
- Replacements SHORT (similar length to original, max ~15 chars), visually plausible on the sign.
- Each entry must have UNIQUE replacements. Do NOT reuse texts across entries.
- Same script/style as original where possible (except "random" which deliberately differs).

**Output JSON ONLY** (no explanation):
{{"texts": [{{"original_text": "...", "text_location": "...", "attacks": {{"similar": "...", "random": "...", "adversarial": "..."}}}}]}}"""
    return prompt


async def process_single_image(provider, image_path, original_filename, clean_img_rel_path, location_info=None):
    """
    Process a single image to generate attacks using the provider.
    """
    prompt = build_attack_prompt(location_info)

    result = await provider.analyze_image_async(
        image_path=Path(image_path),
        prompt=prompt,
        json_mode=True,  # Provider handles Thinking models automatically
    )

    if result.success and result.content:
        try:
            attack_data = json.loads(result.content)

            # Support new multi-text format {"texts": [...]}
            texts_list = attack_data.get("texts", None)

            if texts_list is not None:
                # New format: array of text entries
                valid_texts = []
                for t in texts_list:
                    attacks = t.get("attacks", {})
                    if attacks and t.get("original_text"):
                        valid_texts.append(
                            {
                                "original_text": t.get("original_text", ""),
                                "text_location": t.get("text_location", "in the image"),
                                "attacks": attacks,
                            }
                        )

                if not valid_texts:
                    return None

                return {
                    "original_filename": original_filename,
                    "clean_image_path": clean_img_rel_path,
                    "image_path": image_path,
                    "texts": valid_texts,
                }
            else:
                # Legacy single-text format for backward compatibility
                attacks = attack_data.get("attacks", {})
                if not attacks:
                    return None

                return {
                    "original_filename": original_filename,
                    "clean_image_path": clean_img_rel_path,
                    "image_path": image_path,
                    "texts": [
                        {
                            "original_text": attack_data.get("original_text", ""),
                            "text_location": attack_data.get("text_location", "in the image"),
                            "attacks": attacks,
                        }
                    ],
                }
        except json.JSONDecodeError:
            pass

    return None


async def main_async():
    args = parse_args()

    # ==========================================
    # Initialize Provider based on --provider
    # ==========================================
    if args.provider == "gemini":
        from data_collector.llm_provider import GeminiProvider

        model_name = args.model or "gemini-3.1-flash-lite-preview"
        api_key = args.api_key  # GeminiProvider will fall back to GEMINI_API_KEY env var
        provider = GeminiProvider(
            model_name=model_name,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.9,
        )
        print(f"[Provider] Gemini | Model: {model_name}")
    else:
        from data_collector.llm_provider import OpenAICompatibleProvider

        model_name = args.model or "Qwen/Qwen3-VL-30B-A3B-Thinking"
        api_key = args.api_key or "qwen-local-key"
        provider = OpenAICompatibleProvider(
            model_name=model_name,
            base_url=args.api_base,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.9,
        )
        print(f"[Provider] OpenAI-compatible | Model: {model_name} | Base: {args.api_base}")

    if not provider.is_available():
        print("Error: LLM Provider is not available. Check your API connection / API key.")
        return

    # ==========================================
    # Load Clean Metadata
    # ==========================================
    print(f"Loading metadata from {args.clean_meta}...")
    clean_entries = []
    with open(args.clean_meta, encoding="utf-8") as f:
        for line in f:
            clean_entries.append(json.loads(line))

    print(f"Found {len(clean_entries)} images in metadata.")

    # ==========================================
    # Resume Support: Load already-processed filenames
    # ==========================================
    already_done = set()
    output_path = args.output
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    fname = entry.get("original_filename")
                    if fname:
                        already_done.add(fname)
                except (json.JSONDecodeError, KeyError):
                    pass
        if already_done:
            print(f"Resuming: {len(already_done)} images already processed, will skip.")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # ==========================================
    # Process Loop with Semaphore
    # ==========================================
    tasks = []
    skipped_count = 0
    skipped_done = 0

    sem = asyncio.Semaphore(args.concurrency)

    async def sem_task(entry):
        nonlocal skipped_count, skipped_done
        async with sem:
            # Construct original image path
            fname = entry.get("filename")
            if not fname:
                skipped_count += 1
                return None

            # Skip already processed (resume support)
            if fname in already_done:
                skipped_done += 1
                return None

            # The clean image output filename
            clean_rel_path = entry.get("output_filename")

            original_path = os.path.join(args.original_dir, fname)
            if not os.path.exists(original_path):
                # Try finding with extensions
                found = False
                for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                    if os.path.exists(original_path + ext):
                        original_path = original_path + ext
                        found = True
                        break

                if not found:
                    skipped_count += 1
                    return None

            # Extract location info from metadata
            location_info = {
                "city": entry.get("city", ""),
                "county": entry.get("county", ""),
                "province": entry.get("province", entry.get("state", "")),
                "country": entry.get("country", ""),
            }

            return await process_single_image(
                provider, original_path, fname, clean_rel_path, location_info=location_info
            )

    for i, entry in enumerate(clean_entries):
        if args.limit > 0 and i >= args.limit:
            break
        tasks.append(sem_task(entry))

    print(f"Processing {len(tasks)} images (concurrency={args.concurrency})...")

    # Execute with progress bar and incremental save (append mode for resume)
    completed_tasks = []

    with open(output_path, "a", encoding="utf-8") as f_out:
        for f in tqdm.as_completed(tasks, total=len(tasks)):
            res = await f
            if res:
                completed_tasks.append(res)
                f_out.write(json.dumps(res) + "\n")
                f_out.flush()

    # Summary
    print("\n--- Summary ---")
    print(f"Total in metadata: {len(clean_entries)}")
    print(f"Already processed (skipped): {skipped_done}")
    print(f"Files not found (skipped): {skipped_count}")
    print(f"LLM returned empty: {len(tasks) - skipped_count - skipped_done - len(completed_tasks)}")
    print(f"Successful attacks (this run): {len(completed_tasks)}")
    print(f"Total in output file: {len(already_done) + len(completed_tasks)}")
    print(f"Saved to {output_path}")
    print("Done.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
