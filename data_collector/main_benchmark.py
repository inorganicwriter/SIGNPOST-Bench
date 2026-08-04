import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import random
import threading
from pathlib import Path
from queue import Empty, Queue

from data_collector.comfy_client import ComfyClient
from data_collector.utils import load_workflow_api

COLLECTOR_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

TARGET_MEGAPIXELS = 1.0  # default, will be overridden by args

# Key Node IDs (must match image_qwen_edit.json)
NODE_ID_EDIT_LOAD_IMAGE = "78"
NODE_ID_EDIT_PROMPT = "435"  # PrimitiveStringMultiline → feeds into TextEncodeQwenImageEditPlus
NODE_ID_EDIT_KSAMPLER = "433:3"
NODE_ID_EDIT_QUALITY_MODE = "433:443"  # PrimitiveBoolean: false=quality(20step+CFG4), true=fast(LoRA+4step)
NODE_ID_EDIT_SCALE = "433:117"  # ImageScaleToTotalPixels — match output megapixels

# Key Node IDs for upscale workflow (must match upscale_workflow.json)
NODE_ID_UPSCALE_LOAD_IMAGE = "1"
NODE_ID_UPSCALE_MODEL = "2"
NODE_ID_UPSCALE_MEGAPIXELS = "4"


def generate_upscale_with_comfy(client, workflow_template, input_image_path, model_name, target_mp, seed=None):
    """
    Upscale an image using a ComfyUI upscale model (e.g., RealESRGAN).
    Returns: (image_data, (width, height)) or (None, None) on failure.
    """
    comfy_filename = client.upload_image(input_image_path)
    if not comfy_filename:
        print(f"    -> Upscale upload failed for {input_image_path}")
        return None, None

    workflow = copy.deepcopy(workflow_template)

    if NODE_ID_UPSCALE_LOAD_IMAGE in workflow:
        workflow[NODE_ID_UPSCALE_LOAD_IMAGE]["inputs"]["image"] = comfy_filename

    if NODE_ID_UPSCALE_MODEL in workflow:
        workflow[NODE_ID_UPSCALE_MODEL]["inputs"]["model_name"] = model_name

    if NODE_ID_UPSCALE_MEGAPIXELS in workflow:
        workflow[NODE_ID_UPSCALE_MEGAPIXELS]["inputs"]["megapixels"] = target_mp

    if seed is None:
        seed = random.randint(1, 10**14)

    prompt_res = client.queue_prompt(workflow)
    if not prompt_res:
        print("    -> Upscale queue failed.")
        return None, None

    prompt_id = prompt_res["prompt_id"]

    if not client.wait_for_completion(prompt_id):
        print("    -> Upscale timeout/error.")
        return None, None

    history = client.get_history(prompt_id)
    if not history:
        return None, None

    history_data = history[prompt_id]
    for node_id in history_data["outputs"]:
        node_output = history_data["outputs"][node_id]
        if "images" in node_output:
            for image in node_output["images"]:
                image_data = client.get_image(image["filename"], image["subfolder"], image["type"])
                if image_data:
                    import io as _io

                    from PIL import Image

                    img = Image.open(_io.BytesIO(image_data))
                    return image_data, img.size

    return None, None


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Generator (LLM + ComfyUI)")
    parser.add_argument(
        "--attack-file", type=str, required=True, help="Path to attacks.jsonl (from generate_attacks.py)"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save final benchmark images")
    parser.add_argument(
        "--comfy-server",
        type=str,
        default=None,
        help="ComfyUI server address (single server mode, e.g. 127.0.0.1:8188)",
    )
    parser.add_argument(
        "--comfy-servers",
        type=str,
        default=None,
        help="Comma-separated ComfyUI server addresses for multi-GPU parallel mode. "
        "E.g.: '10.0.0.1:8188,10.0.0.2:8188,10.0.0.3:8188,10.0.0.4:8188'",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images processed")
    parser.add_argument(
        "--megapixels", type=float, default=1.0, help="Target megapixels for output images (default: 1.0 ~ 1MB)"
    )
    parser.add_argument(
        "--upscale-workflow",
        type=str,
        default=None,
        help="Path to ComfyUI upscale workflow JSON (default: data_collector/upscale_workflow.json)",
    )
    parser.add_argument(
        "--upscale-model",
        type=str,
        default="4x-UltraSharp.pth",
        help="ComfyUI upscale model name (default: 4x-UltraSharp.pth)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variants to generate: Original,Blank,Similar,Random,Adversarial,all",
    )
    parser.add_argument(
        "--restore-original",
        action="store_true",
        help="Use Qwen Image Edit to restore/enhance Original instead of upscale model",
    )
    return parser.parse_args()


RESTORE_PROMPT = "Enhance this image to high quality. Sharpen fine details, improve clarity, restore textures, and reduce noise. Do not alter the content or composition in any way."


def generate_image_with_comfy(client, workflow_template, input_image_path, prompt, seed=None):
    """
    Helper function to run the ComfyUI workflow for a single image generation.
    Returns: image_data (bytes) or None if failed.
    """
    # 1. Upload Input Image
    comfy_filename = client.upload_image(input_image_path)
    if not comfy_filename:
        print(f"    -> Upload failed for {input_image_path}")
        return None, None

    # 2. Configure Workflow (deep copy to avoid mutating template)
    workflow = copy.deepcopy(workflow_template)

    if NODE_ID_EDIT_LOAD_IMAGE in workflow:
        workflow[NODE_ID_EDIT_LOAD_IMAGE]["inputs"]["image"] = comfy_filename

    if NODE_ID_EDIT_PROMPT in workflow:
        workflow[NODE_ID_EDIT_PROMPT]["inputs"]["value"] = prompt

    if NODE_ID_EDIT_QUALITY_MODE in workflow:
        workflow[NODE_ID_EDIT_QUALITY_MODE]["inputs"]["value"] = False

    if NODE_ID_EDIT_SCALE in workflow:
        workflow[NODE_ID_EDIT_SCALE]["inputs"]["megapixels"] = TARGET_MEGAPIXELS

    if seed is None:
        seed = random.randint(1, 10**14)

    if NODE_ID_EDIT_KSAMPLER in workflow:
        workflow[NODE_ID_EDIT_KSAMPLER]["inputs"]["seed"] = seed

    # 3. Queue Prompt
    prompt_res = client.queue_prompt(workflow)
    if not prompt_res:
        print("    -> Queue failed.")
        return None, None

    prompt_id = prompt_res["prompt_id"]

    # 4. Wait for Completion
    if not client.wait_for_completion(prompt_id):
        print("    -> Timeout/Error.")
        return None, None

    # 5. Retrieve Image
    history = client.get_history(prompt_id)
    if not history:
        return None, None

    history_data = history[prompt_id]
    for node_id in history_data["outputs"]:
        node_output = history_data["outputs"][node_id]
        if "images" in node_output:
            for image in node_output["images"]:
                image_data = client.get_image(image["filename"], image["subfolder"], image["type"])
                if image_data:
                    return image_data, seed  # Return data and used seed

    return None, None


def _check_variants_exist(base_name, texts_list, output_dir, variants):
    """Check if all requested variant images already exist (resume support)."""
    if "Original" in variants:
        orig_path = os.path.join(output_dir, "Original", f"{base_name}.png")
        if not os.path.exists(orig_path):
            return False

    if "Blank" in variants:
        blank_save_path = os.path.join(output_dir, "Blank", f"{base_name}_Blank.png")
        if not os.path.exists(blank_save_path):
            return False

    for text_entry in texts_list:
        attack_dict = text_entry.get("attacks", {})
        for attack_type, attack_text in attack_dict.items():
            variant_name = attack_type.capitalize()
            if variant_name not in variants:
                continue
            safe_text = "".join([c for c in attack_text if c.isalnum() or c in (" ", "_", "-")]).strip()
            subdir = variant_name
            candidate = f"{base_name}_{attack_type}_{safe_text}.png"
            if len(candidate.encode("utf-8")) > 255:
                max_text_len = 255 - len(f"{base_name}_{attack_type}_.png".encode())
                safe_text = safe_text[: max(10, max_text_len)]
                candidate = f"{base_name}_{attack_type}_{safe_text}.png"
            check_path = os.path.join(output_dir, subdir, candidate)
            if not os.path.exists(check_path):
                return False
    return True


def build_blank_prompt(texts_list):
    """Build the blank (text removal) prompt."""
    base = "Reconstruct the background surface naturally to match the surrounding area. Leave the edited region clean and seamless with no ghosting, smudging, or residual marks. The inpainted area must blend seamlessly with the surrounding surface. Do not alter anything else in the image."
    removals = []
    for t in texts_list:
        orig_text = t.get("original_text", "")
        loc = t.get("text_location", "in the image")
        removals.append(f'Remove the text "{orig_text}" located {loc}')
    if len(removals) == 1:
        return f"{removals[0]}. {base}"
    else:
        return ". ".join(removals) + ". " + base


def build_attack_prompt(texts_list, attack_type):
    """Build the attack (text replacement) prompt. Returns (prompt, all_attack_texts) or (None, None)."""
    replacements = []
    all_attack_texts = []
    base = "Match the original font style, size, and color. Render the new text clearly and legibly without distortion or blurring. The edited area must blend seamlessly with the surrounding surface. Do not alter anything else in the image."
    for text_entry in texts_list:
        attack_dict = text_entry.get("attacks", {})
        attack_text = attack_dict.get(attack_type)
        if not attack_text:
            continue
        original_text = text_entry.get("original_text", "")
        text_location = text_entry.get("text_location", "in the image")
        replacements.append(f'Replace the text "{original_text}" located {text_location} with "{attack_text}"')
        all_attack_texts.append(attack_text)

    if not replacements:
        return None, None

    if len(replacements) == 1:
        prompt = f"{replacements[0]}. {base}"
    else:
        prompt = ". ".join(replacements) + ". " + base

    return prompt, all_attack_texts


def process_entry(
    entry,
    idx,
    total,
    client,
    workflow_template,
    upscale_workflow_template,
    upscale_model,
    output_dir,
    variants,
    restore_original,
    meta_lock,
    meta_file_path,
):
    """Process a single attack entry: upscale original and/or generate specified variants."""
    original_filename = entry.get("original_filename")
    base_name = os.path.splitext(original_filename)[0]
    worker_tag = f"[{client.server_address}]"

    # Locate Original Clean Image
    clean_img_path = entry.get("image_path")
    if not clean_img_path or not os.path.exists(clean_img_path):
        from config import CLEAN_IMAGES_DIR

        clean_img_path = str(CLEAN_IMAGES_DIR / entry.get("clean_image_path", ""))
        if not os.path.exists(clean_img_path):
            print(f"{worker_tag} Warning: Clean image not found for {original_filename}. Skipping.")
            return False

    # Normalize to multi-text format
    texts_list = entry.get("texts", None)
    if texts_list is None:
        attack_dict = entry.get("attacks", {})
        text_location = entry.get("text_location", "in the image")
        if not attack_dict:
            return False
        texts_list = [
            {"original_text": entry.get("original_text", ""), "text_location": text_location, "attacks": attack_dict}
        ]

    if not texts_list:
        return False

    num_texts = len(texts_list)

    # Resume check — only check requested variants
    if _check_variants_exist(base_name, texts_list, output_dir, variants):
        print(f"{worker_tag} [{idx + 1}/{total}] {original_filename} - Already complete, skipping.")
        return True

    print(f"{worker_tag} [{idx + 1}/{total}] Processing {original_filename} ({num_texts} text(s))...")

    need_original = "Original" in variants
    need_edit = variants & {"Blank", "Similar", "Random", "Adversarial"}

    orig_save_path = os.path.join(output_dir, "Original", f"{base_name}.png")

    # ============================================================
    # Step 0: Original — upscale model or Qwen Edit restore
    # ============================================================
    if need_original:
        if restore_original:
            # Use Qwen Image Edit with restoration prompt
            upscaled_data, upscaled_seed = generate_image_with_comfy(
                client, workflow_template, clean_img_path, RESTORE_PROMPT
            )
            if not upscaled_data:
                print(f"{worker_tag}   -> Failed to restore {original_filename}. Skipping.")
                return False
            upscaled_w = upscaled_h = "?"
            method = "Qwen Edit restore"
        else:
            result = generate_upscale_with_comfy(
                client, upscale_workflow_template, clean_img_path, upscale_model, TARGET_MEGAPIXELS
            )
            if not result or not result[0]:
                print(f"{worker_tag}   -> Failed to upscale {original_filename}. Skipping.")
                return False
            upscaled_data, (upscaled_w, upscaled_h) = result
            method = upscale_model

        os.makedirs(os.path.dirname(orig_save_path), exist_ok=True)
        with open(orig_save_path, "wb") as f:
            f.write(upscaled_data)
        print(
            f"{worker_tag}   -> Saved Original ({upscaled_w}x{upscaled_h}) via {method}: Original/{os.path.basename(orig_save_path)}"
        )
    elif need_edit:
        if not os.path.exists(orig_save_path):
            print(f"{worker_tag}   -> Original not found at {orig_save_path}. Run --variants Original first. Skipping.")
            return False

    if not need_edit:
        return True

    # ============================================================
    # Step 1-N: Edit variants — read Original from disk as input
    # ============================================================
    try:
        # Phase 1: Generate BLANK Image
        if "Blank" in variants:
            blank_prompt = build_blank_prompt(texts_list)
            blank_image_data, blank_seed = generate_image_with_comfy(
                client, workflow_template, orig_save_path, blank_prompt
            )

            if not blank_image_data:
                print(f"{worker_tag}   -> Failed to generate Blank image. Skipping.")
                return False

            blank_save_dir = os.path.join(output_dir, "Blank")
            os.makedirs(blank_save_dir, exist_ok=True)
            blank_save_name = f"{base_name}_Blank.png"
            blank_save_path = os.path.join(blank_save_dir, blank_save_name)

            with open(blank_save_path, "wb") as f:
                f.write(blank_image_data)
            print(f"{worker_tag}   -> Saved Blank: Blank/{blank_save_name}")

            meta_entry = {
                "filename": blank_save_name,
                "original_source": original_filename,
                "source_image_used": clean_img_path,
                "injected_text": "",
                "attack_type": "Blank",
                "prompt_used": blank_prompt,
                "seed": blank_seed,
                "num_texts_removed": num_texts,
            }
            with meta_lock:
                with open(meta_file_path, "a", encoding="utf-8") as meta_f:
                    meta_f.write(json.dumps(meta_entry) + "\n")

        # Phase 2: Generate Attacks
        for attack_type in ["similar", "random", "adversarial"]:
            variant_name = attack_type.capitalize()
            if variant_name not in variants:
                continue

            prompt, all_attack_texts = build_attack_prompt(texts_list, attack_type)
            if not prompt:
                continue

            combined_text_label = " + ".join(all_attack_texts)
            print(f"{worker_tag}   -> Generating {attack_type}: '{combined_text_label}'")

            attack_image_data, attack_seed = generate_image_with_comfy(
                client, workflow_template, orig_save_path, prompt
            )

            if not attack_image_data:
                print(f"{worker_tag}   -> Failed to generate {attack_type}.")
                continue

            subdir = variant_name
            save_dir = os.path.join(output_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)

            safe_text = "".join([c for c in combined_text_label if c.isalnum() or c in (" ", "_", "-")]).strip()
            save_name = f"{base_name}_{attack_type}_{safe_text}.png"
            if len(save_name.encode("utf-8")) > 255:
                max_text_len = 255 - len(f"{base_name}_{attack_type}_.png".encode())
                safe_text = safe_text[: max(10, max_text_len)]
                save_name = f"{base_name}_{attack_type}_{safe_text}.png"
            save_path = os.path.join(save_dir, save_name)

            with open(save_path, "wb") as f:
                f.write(attack_image_data)
            print(f"{worker_tag}   -> Saved: {subdir}/{save_name}")

            meta_entry = {
                "filename": save_name,
                "original_source": original_filename,
                "clean_source": clean_img_path,
                "injected_texts": all_attack_texts,
                "attack_type": attack_type,
                "prompt_used": prompt,
                "seed": attack_seed,
                "num_texts_replaced": len(all_attack_texts),
            }
            with meta_lock:
                with open(meta_file_path, "a", encoding="utf-8") as meta_f:
                    meta_f.write(json.dumps(meta_entry) + "\n")
    except Exception:
        raise

    return True


def worker_thread(
    worker_id,
    server_address,
    task_queue,
    workflow_template,
    upscale_workflow_template,
    upscale_model,
    output_dir,
    variants,
    restore_original,
    meta_lock,
    meta_file_path,
    stats,
    total_tasks,
):
    """Worker thread: connects to one ComfyUI instance and processes tasks from the queue."""
    print(f"[Worker {worker_id}] Connecting to ComfyUI at {server_address}...")
    client = ComfyClient(server_address)
    if not client.connect():
        print(f"[Worker {worker_id}] Failed to connect to {server_address}!")
        return

    print(f"[Worker {worker_id}] Connected to {server_address}")

    while True:
        try:
            idx, entry = task_queue.get_nowait()
        except Empty:
            break

        try:
            success = process_entry(
                entry,
                idx,
                total_tasks,
                client,
                workflow_template,
                upscale_workflow_template,
                upscale_model,
                output_dir,
                variants,
                restore_original,
                meta_lock,
                meta_file_path,
            )
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
        except Exception as e:
            print(f"[Worker {worker_id}] Error processing entry {idx}: {e}")
            stats["failed"] += 1
            try:
                client.close()
                client = ComfyClient(server_address)
                client.connect()
            except Exception:
                pass
        finally:
            task_queue.task_done()

    client.close()
    print(f"[Worker {worker_id}] Done.")


def main():
    args = parse_args()
    global TARGET_MEGAPIXELS
    TARGET_MEGAPIXELS = args.megapixels
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse variants
    if args.variants.lower() == "all":
        variants = {"Original", "Blank", "Similar", "Random", "Adversarial"}
    else:
        variants = {v.strip().capitalize() for v in args.variants.split(",") if v.strip()}

    # Determine server list
    if args.comfy_servers:
        servers = [s.strip() for s in args.comfy_servers.split(",") if s.strip()]
    elif args.comfy_server:
        servers = [args.comfy_server]
    else:
        from config import COMFY_SERVER

        servers = [COMFY_SERVER] if COMFY_SERVER else None
    if not servers:
        print("Error: --comfy-server or --comfy-servers must be provided, or set COMFY_SERVER in .env")
        return

    num_workers = len(servers)
    print("=== SIGNPOST-Bench Image Synthesis ===")
    print(f"Workers: {num_workers} ({', '.join(servers)})")
    print(f"Upscale model: {args.upscale_model}")
    print(f"Variants: {', '.join(sorted(variants))}")
    if args.restore_original:
        print("Original method: Qwen Image Edit restore")

    # Load Edit Workflow Template
    workflow_path = COLLECTOR_DIR / "image_qwen_edit.json"
    if not os.path.exists(workflow_path):
        print(f"Error: Edit workflow file not found at {workflow_path}")
        return

    workflow_template = load_workflow_api(workflow_path)
    if not workflow_template:
        print("Error: Failed to load edit workflow template.")
        return

    # Only needed when generating Original through the upscale workflow.
    needs_upscale_workflow = "Original" in variants and not args.restore_original
    if needs_upscale_workflow:
        upscale_workflow_path = Path(args.upscale_workflow) if args.upscale_workflow else COLLECTOR_DIR / "upscale_workflow.json"
        if not os.path.exists(upscale_workflow_path):
            print(f"Error: Upscale workflow file not found at {upscale_workflow_path}")
            return

        upscale_workflow_template = load_workflow_api(upscale_workflow_path)
        if not upscale_workflow_template:
            print("Error: Failed to load upscale workflow template.")
            return
    else:
        upscale_workflow_template = None

    # Load Attacks
    print(f"Reading attacks from {args.attack_file}...")
    attacks = []
    with open(args.attack_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                attacks.append(json.loads(line))

    if args.limit > 0:
        attacks = attacks[: args.limit]

    total = len(attacks)
    print(f"Found {total} entries to process with {num_workers} worker(s).")

    # Build meta file path — use variant-specific suffix when running split mode
    if args.variants.lower() == "all":
        meta_file_path = os.path.join(args.output_dir, "benchmark_meta.jsonl")
    else:
        variant_tag = args.variants.replace(",", "_").replace(" ", "").lower()
        meta_file_path = os.path.join(args.output_dir, f"benchmark_meta__{variant_tag}.jsonl")

    if num_workers == 1:
        # ==========================================
        # Single-worker mode (original behavior)
        # ==========================================
        print(f"\nSingle-worker mode: {servers[0]}")
        client = ComfyClient(servers[0])
        if not client.connect():
            print("Failed to connect to ComfyUI. Exiting.")
            return

        meta_lock = threading.Lock()
        processed = 0
        for i, entry in enumerate(attacks):
            success = process_entry(
                entry,
                i,
                total,
                client,
                workflow_template,
                upscale_workflow_template,
                args.upscale_model,
                args.output_dir,
                variants,
                args.restore_original,
                meta_lock,
                meta_file_path,
            )
            if success:
                processed += 1

        client.close()
        print(f"\nComplete. {processed}/{total} entries processed.")

    else:
        # ==========================================
        # Multi-worker parallel mode
        # ==========================================
        print(f"\nMulti-worker mode: {num_workers} GPUs")

        # Build task queue
        task_queue = Queue()
        for i, entry in enumerate(attacks):
            task_queue.put((i, entry))

        meta_lock = threading.Lock()

        # Per-worker stats
        worker_stats = [{"success": 0, "failed": 0} for _ in range(num_workers)]

        # Launch worker threads
        threads = []
        for w_id, server in enumerate(servers):
            t = threading.Thread(
                target=worker_thread,
                args=(
                    w_id,
                    server,
                    task_queue,
                    workflow_template,
                    upscale_workflow_template,
                    args.upscale_model,
                    args.output_dir,
                    variants,
                    args.restore_original,
                    meta_lock,
                    meta_file_path,
                    worker_stats[w_id],
                    total,
                ),
                daemon=True,
            )
            threads.append(t)
            t.start()

        # Wait for all tasks to complete
        task_queue.join()

        # Wait for threads to finish
        for t in threads:
            t.join(timeout=10)

        # Print summary
        total_success = sum(s["success"] for s in worker_stats)
        total_failed = sum(s["failed"] for s in worker_stats)

        print(f"\n{'=' * 50}")
        print("  Synthesis Complete!")
        print(f"  Total: {total} | Success: {total_success} | Failed: {total_failed}")
        print(f"{'=' * 50}")
        for w_id, (server, stats) in enumerate(zip(servers, worker_stats, strict=False)):
            print(f"  Worker {w_id} ({server}): {stats['success']} success, {stats['failed']} failed")

    print(f"\nMetadata saved to {meta_file_path}")


if __name__ == "__main__":
    main()
