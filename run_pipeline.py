import argparse
import os
import subprocess
import sys
from pathlib import Path

# ================= Configuration =================
# Import centralized config (paths resolved from env vars with defaults)
from config import (
    CODE_DIR,
    COMFY_SERVER,
    COMFY_SERVERS,
    DEFAULT_MODEL,
    LOCAL_API_BASE,
    RAW_DATA_ROOT,
    get_output_path,
)
from config import (
    DATA_ROOT as TRIG_BENCH_ROOT,
)

# Import unified model registry from api_client
# This avoids duplicating model definitions across files
try:
    sys.path.insert(0, str(CODE_DIR))
    from evaluation.api_client import MODEL_REGISTRY, PROVIDER_CONFIGS

    _registry_loaded = True
except ImportError:
    _registry_loaded = False
    MODEL_REGISTRY = {}
    PROVIDER_CONFIGS = {}


def resolve_model(model_arg):
    """
    Resolve model short name to (model_path, short_name, api_base).
    Uses the unified MODEL_REGISTRY from evaluation/api_client.py.
    Falls back to local vLLM for unknown names.
    """
    if model_arg.endswith("-vertex"):
        raise ValueError(
            "Legacy Vertex model aliases are no longer supported. "
            "Use supported short names such as 'gemini-2.5-flash' or 'gemini-3.1-pro'."
        )
    if model_arg in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_arg]
        provider = entry.get("provider", "sponsor")
        provider_cfg = PROVIDER_CONFIGS.get(provider, {})
        api_base = provider_cfg.get("api_base", LOCAL_API_BASE)
        return entry["model"], model_arg, api_base
    # If full path given, assume sponsor gateway
    base = model_arg.rstrip("/").split("/")[-1].lower()
    provider_cfg = PROVIDER_CONFIGS.get("sponsor", {})
    api_base = provider_cfg.get("api_base", LOCAL_API_BASE)
    return model_arg, base, api_base


def parse_args():
    parser = argparse.ArgumentParser(description="Run SIGNPOST-Bench Pipeline for a specific dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset, or 'all' to run all datasets. Available: im2gps3k, yfcc4k, googlesv, baidusv",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "attack_gen", "synthesize", "evaluate"],
        default="all",
        help="Run a specific stage. 'attack_gen' includes filtering, metadata, and LLM generation. Default: all",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip OCR filtering step (use when filtered_images already exists but raw images are gone)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="API Key (default: None, uses env vars for cloud providers)"
    )
    parser.add_argument("--api-base", type=str, default=None, help="Override API base URL")
    parser.add_argument(
        "--raw-img-dir",
        type=str,
        default=None,
        help="Override raw image directory (default: auto-derived from dataset name)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model for evaluation (short name or full path). Short names: " + ", ".join(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--attack-provider",
        type=str,
        default="gemini",
        choices=["openai", "gemini"],
        help="LLM provider for attack generation: 'gemini' or 'openai' (default: gemini)",
    )
    parser.add_argument(
        "--attack-model", type=str, default=None, help="Model for attack generation (default: auto per provider)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Concurrent API requests for attack generation (default: 10)"
    )
    parser.add_argument(
        "--upscale-model",
        type=str,
        default="4x-UltraSharp.pth",
        help="ComfyUI upscale model for Original variant generation (default: 4x-UltraSharp.pth)",
    )
    return parser.parse_args()


# Dataset-specific image directory names
# When the image subdirectory doesn't match the dataset name
IMAGE_DIR_OVERRIDES = {
    "im2gps3k": "im2gps3ktest",
    # Add more here as needed, e.g.:
    # "streetview": "images",
}


def get_paths(dataset_name, raw_img_dir_override=None):
    # Derived Paths based on dataset name
    # Supports override for datasets with non-standard directory structures

    work_dir = TRIG_BENCH_ROOT / dataset_name

    if raw_img_dir_override:
        raw_img_dir = Path(raw_img_dir_override)
    elif dataset_name == "googlesv":
        # GoogleSV uses sampled images in the SIGNPOST-Bench work directory
        raw_img_dir = work_dir / "sampled_images"
    elif dataset_name == "baidusv":
        # BaiduSV uses sampled & cropped images in the SIGNPOST-Bench work directory
        raw_img_dir = work_dir / "sampled_images"
    else:
        img_subdir = IMAGE_DIR_OVERRIDES.get(dataset_name, dataset_name)
        raw_img_dir = RAW_DATA_ROOT / dataset_name / img_subdir

    # Metadata CSV location
    if dataset_name == "googlesv":
        raw_meta_csv = work_dir / "googlesv_metadata_address.csv"
    elif dataset_name == "baidusv":
        raw_meta_csv = work_dir / "baidusv_metadata.csv"
    else:
        raw_meta_csv = RAW_DATA_ROOT / dataset_name / f"{dataset_name}_metadata_address.csv"

    return {
        "raw_img_dir": raw_img_dir,
        "raw_meta_csv": raw_meta_csv,
        "work_dir": work_dir,
        "metadata_dir": work_dir / "metadata",
        "images_dir": work_dir / "images",
        "results_dir": work_dir / "results",
        "attacks_file": work_dir / "attacks.jsonl",
        "dataset_name": dataset_name,
    }


def run_step(step_name, command, cwd=CODE_DIR):
    print(f"\n{'=' * 10} Step: {step_name} {'=' * 10}")
    print(f"Running: {' '.join(str(x) for x in command)}")
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {step_name}: {e}")
        sys.exit(1)


def ensure_dirs(paths):
    print(f"Creating directories in {paths['work_dir']}...")
    os.makedirs(paths["metadata_dir"], exist_ok=True)
    os.makedirs(paths["images_dir"], exist_ok=True)
    os.makedirs(paths["results_dir"], exist_ok=True)


ALL_DATASETS = ["im2gps3k", "yfcc4k", "googlesv", "baidusv"]


def run_single_dataset(args, dataset_name):
    """Run the pipeline for a single dataset."""
    paths = get_paths(dataset_name, raw_img_dir_override=args.raw_img_dir)

    # Verify input exists (Only strictly enforced for attack_gen stage when not skipping filter)
    if args.stage in ["all", "attack_gen"] and not args.skip_filter and not paths["raw_img_dir"].exists():
        print(f"Error: Raw image directory not found: {paths['raw_img_dir']}")
        print("  Hint: If you already have filtered_images, use --skip-filter to skip the OCR step.")
        sys.exit(1)

    ensure_dirs(paths)

    # Resolve model
    try:
        if args.model:
            model_path, model_short, model_api_base = resolve_model(args.model)
        else:
            model_path, model_short, model_api_base = resolve_model(DEFAULT_MODEL)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    # Allow CLI override of api-base
    api_base = args.api_base if args.api_base else model_api_base

    # Derived file paths helper
    filtered_img_dir = paths["work_dir"] / "filtered_images"
    clean_meta = paths["metadata_dir"] / f"{paths['dataset_name']}_clean_meta.jsonl"
    gt_tsv = paths["metadata_dir"] / f"{paths['dataset_name']}_gt.tsv"
    bench_meta = paths["images_dir"] / "benchmark_meta.jsonl"

    # ================= Stage 1: Attack Generation (Filter + Metadata + LLM) =================
    if args.stage in ["all", "attack_gen"]:
        print(">>> Stage: Attack Generation (Filter -> Metadata -> LLM)")

        # Step 0: Filter Images (can be skipped with --skip-filter)
        if args.skip_filter:
            if filtered_img_dir.exists():
                print(f">>> --skip-filter: Using existing filtered_images at {filtered_img_dir}")
                print(f"    ({len(list(filtered_img_dir.iterdir()))} files found)")
            else:
                print(f"Error: --skip-filter specified but {filtered_img_dir} does not exist!")
                sys.exit(1)
        else:
            run_step(
                "Filter Images (OCR)",
                [
                    sys.executable,
                    "data_collector/filter_images.py",
                    "--input-dir",
                    str(paths["raw_img_dir"]),
                    "--output-dir",
                    str(filtered_img_dir),
                    "--gpu",
                ],
            )

        # Step 1: Prepare Metadata (always needed — reads from CSV, not images)
        if clean_meta.exists() and args.skip_filter:
            print(f">>> Metadata already exists at {clean_meta}, skipping regeneration.")
        else:
            run_step(
                "Prepare Metadata",
                [
                    sys.executable,
                    "data_collector/convert_metadata.py",
                    "--csv",
                    str(paths["raw_meta_csv"]),
                    "--out-dir",
                    str(paths["metadata_dir"]),
                    "--dataset-name",
                    paths["dataset_name"],
                ],
            )

        # Step 2: Generate Attacks
        print(">>> Generating Attacks with LLM...")
        if not filtered_img_dir.exists():
            print(f"Warning: Filtered images dir {filtered_img_dir} not found.")

        # Resolve attack model: use --attack-model if given, else --model, else provider default
        attack_model = args.attack_model or (model_path if args.attack_provider == "openai" else None)

        attack_cmd = [
            sys.executable,
            "data_collector/generate_attacks.py",
            "--clean-meta",
            str(clean_meta),
            "--original-dir",
            str(filtered_img_dir),
            "--output",
            str(paths["attacks_file"]),
            "--provider",
            args.attack_provider,
            "--concurrency",
            str(args.concurrency),
        ]
        if attack_model:
            attack_cmd.extend(["--model", attack_model])
        if args.attack_provider == "openai":
            attack_cmd.extend(["--api-base", api_base])
        if args.api_key:
            attack_cmd.extend(["--api-key", args.api_key])
        run_step("Generate Attacks", attack_cmd)

    # ================= Stage 2: Synthesis =================
    if args.stage in ["all", "synthesize"]:
        print(">>> Stage: Image Synthesis (ComfyUI)")

        if not paths["attacks_file"].exists():
            print(f"Error: Attack file {paths['attacks_file']} not found. Run 'attack' stage first.")
            sys.exit(1)

        synth_cmd = [
            sys.executable,
            "data_collector/main_benchmark.py",
            "--attack-file",
            str(paths["attacks_file"]),
            "--output-dir",
            str(paths["images_dir"]),
            "--upscale-model",
            args.upscale_model,
        ]
        # Use multi-GPU if COMFY_SERVERS is configured
        if COMFY_SERVERS:
            synth_cmd.extend(["--comfy-servers", COMFY_SERVERS])
        else:
            synth_cmd.extend(["--comfy-server", COMFY_SERVER])
        run_step("Synthesize Images", synth_cmd)

    # ================= Stage 4: Evaluation =================
    if args.stage in ["all", "evaluate"]:
        print(f">>> Stage: Evaluation (Model: {model_short})")

        # Step 1: Evaluate Original Images for capability reporting.
        blank_result_file = get_output_path(dataset_name, "Blank", model_short)
        original_result_file = get_output_path(dataset_name, "Original", model_short)

        # Build common eval args (api-key only appended when not None)
        def _build_eval_cmd(img_dir, output_file, bench_meta_path=None, baseline_path=None):
            cmd = [
                sys.executable,
                "evaluate.py",
                "--img-dir",
                str(img_dir),
                "--metadata-file",
                str(gt_tsv),
                "--output",
                str(output_file),
                "--model",
                model_short,  # FIX: pass short name, not model_path
                "--api-base",
                api_base,
            ]
            if args.api_key:  # FIX: only pass --api-key when explicitly provided
                cmd.extend(["--api-key", args.api_key])
            if bench_meta_path:
                cmd.extend(["--bench-meta", str(bench_meta_path)])
            if baseline_path and Path(baseline_path).exists():
                cmd.extend(["--baseline", str(baseline_path)])
            return cmd

        blank_img_dir = paths["images_dir"] / "Blank"

        if filtered_img_dir.exists() and filtered_img_dir.is_dir():
            print(f"\n--- Evaluating Original Images ({model_short}) ---")
            run_step(f"Evaluate Original ({model_short})", _build_eval_cmd(filtered_img_dir, original_result_file))
        else:
            print(f"Warning: Filtered images dir {filtered_img_dir} not found, Original evaluation will be skipped.")

        # Step 2: Evaluate Blank Images (baseline for TBS).
        if blank_img_dir.exists() and blank_img_dir.is_dir():
            print(f"\n--- Evaluating Blank Images ({model_short}) ---")
            run_step(
                f"Evaluate Blank ({model_short})",
                _build_eval_cmd(blank_img_dir, blank_result_file, bench_meta_path=bench_meta),
            )
        else:
            print(f"Warning: Blank images dir {blank_img_dir} not found, TBS will be unavailable.")

        # Step 3: Evaluate Attack Images (with Blank baseline for TBS).
        subdirs = ["Adversarial", "Similar", "Random"]

        for subdir in subdirs:
            target_dir = paths["images_dir"] / subdir
            if target_dir.exists() and target_dir.is_dir():
                print(f"\n--- Evaluating Subdirectory: {subdir} ({model_short}) ---")
                result_file = get_output_path(dataset_name, subdir, model_short)
                run_step(
                    f"Evaluate {subdir} ({model_short})",
                    _build_eval_cmd(
                        target_dir, result_file, bench_meta_path=bench_meta, baseline_path=blank_result_file
                    ),
                )
            else:
                print(f"Skipping evaluation for {subdir} (Directory not found)")

    print(f"\nPipeline for '{dataset_name}' (stage='{args.stage}') completed! Results in {paths['results_dir']}")


def main():
    args = parse_args()

    # Support --dataset all to run all datasets sequentially
    if args.dataset.lower() == "all":
        datasets = ALL_DATASETS
        print(f"{'=' * 60}")
        print(f"  Running ALL datasets: {', '.join(datasets)}")
        print(f"  Stage: {args.stage}")
        print(f"{'=' * 60}")
    else:
        datasets = [args.dataset]

    for i, ds in enumerate(datasets):
        if len(datasets) > 1:
            print(f"\n{'#' * 60}")
            print(f"  Dataset [{i + 1}/{len(datasets)}]: {ds}")
            print(f"{'#' * 60}")
        run_single_dataset(args, ds)

    if len(datasets) > 1:
        print(f"\n{'=' * 60}")
        print(f"  All {len(datasets)} datasets completed!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
