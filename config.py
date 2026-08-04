"""
config.py — Centralized configuration for SIGNPOST-Bench.

All paths, API keys, and service URLs are resolved from environment variables,
falling back to sensible defaults.

To customize for your environment, either:
  1. Set environment variables (recommended):
       export SIGNPOST_DATA_ROOT=/path/to/data
       export SPONSOR_API_KEY=sk-xxx
  2. Or create a .env file in the project root (loaded automatically).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Try loading .env file (optional dependency)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
# Project code directory (auto-detected from this file's location)
CODE_DIR = Path(__file__).parent.resolve()

# Data root: where datasets, images, and results are stored.
# Default points to the sibling Data/ folder of this supplement.
# Priority: env var > sibling Data/ > local ./data
DATA_ROOT = Path(
    os.environ.get(
        "SIGNPOST_DATA_ROOT",
        str((CODE_DIR.parent / "Data") if (CODE_DIR.parent / "Data").is_dir() else (CODE_DIR / "data")),
    )
)

# Raw dataset root (original unprocessed datasets, for data collection only)
RAW_DATA_ROOT = Path(os.environ.get("SIGNPOST_RAW_DATA_ROOT", str(DATA_ROOT / "raw")))

# Google Street View source data
GOOGLESV_ROOT = Path(os.environ.get("SIGNPOST_GOOGLESV_ROOT", str(RAW_DATA_ROOT / "GoogleSV")))

# Baidu Street View source data
BAIDUSV_ROOT = Path(os.environ.get("SIGNPOST_BAIDUSV_ROOT", str(RAW_DATA_ROOT / "BaiduSV")))

# Results root directory.
# Default layout keeps results alongside each dataset:
#   data/<dataset>/results/<model>/results_<Variant>_<Model>.jsonl
# If SIGNPOST_RESULTS_DIR is set, it is treated as the root that contains
# per-dataset result folders.
RESULTS_DIR = Path(os.environ.get("SIGNPOST_RESULTS_DIR", str(DATA_ROOT)))
ROOT_RESULTS_DIR = CODE_DIR / "results"

# Analysis / paper artifact roots
ANALYSIS_DIR = CODE_DIR / "analysis"
PAPER_DIR = CODE_DIR / "paper"
EXPERIMENT_OUTPUTS_DIR = ANALYSIS_DIR / "experiment_outputs"
EXPERIMENT_SUBSETS_DIR = ANALYSIS_DIR / "subsets"

# Shared analysis artifacts
GEOCODE_CACHE_FILE = DATA_ROOT / "geocode_cache.json"
TAXONOMY_ANNOTATIONS_FILE = DATA_ROOT / "taxonomy_annotations.csv"
CLEAN_IMAGES_DIR = DATA_ROOT / "clean_images"

# ---------------------------------------------------------------------------
# Dataset Configurations
# ---------------------------------------------------------------------------
DATASETS = {
    "im2gps3k": {
        "images_dir": DATA_ROOT / "im2gps3k" / "images",
        "metadata_file": DATA_ROOT / "im2gps3k" / "metadata" / "im2gps3k_gt.tsv",
        "bench_meta": DATA_ROOT / "im2gps3k" / "images" / "benchmark_meta.jsonl",
        "attacks_file": DATA_ROOT / "im2gps3k" / "attacks.jsonl",
    },
    "yfcc4k": {
        "images_dir": DATA_ROOT / "yfcc4k" / "images",
        "metadata_file": DATA_ROOT / "yfcc4k" / "metadata" / "yfcc4k_gt.tsv",
        "bench_meta": DATA_ROOT / "yfcc4k" / "images" / "benchmark_meta.jsonl",
        "attacks_file": DATA_ROOT / "yfcc4k" / "attacks.jsonl",
    },
    "googlesv": {
        "images_dir": DATA_ROOT / "googlesv" / "images",
        "metadata_file": DATA_ROOT / "googlesv" / "metadata" / "googlesv_gt.tsv",
        "bench_meta": DATA_ROOT / "googlesv" / "images" / "benchmark_meta.jsonl",
        "attacks_file": DATA_ROOT / "googlesv" / "attacks.jsonl",
    },
    "baidusv": {
        "images_dir": DATA_ROOT / "baidusv" / "images",
        "metadata_file": DATA_ROOT / "baidusv" / "metadata" / "baidusv_gt.tsv",
        "bench_meta": DATA_ROOT / "baidusv" / "images" / "benchmark_meta.jsonl",
        "attacks_file": DATA_ROOT / "baidusv" / "attacks.jsonl",
    },
}

# Image variant subdirectories
IMAGE_VARIANTS = ["Original", "Blank", "Similar", "Random", "Adversarial"]

# ---------------------------------------------------------------------------
# Sponsor API Gateway (primary provider for evaluation)
# ---------------------------------------------------------------------------
SPONSOR_API_BASE = os.environ.get("SPONSOR_API_BASE", "")
SPONSOR_API_KEY = os.environ.get("SPONSOR_API_KEY", "")

# Gemini API key for direct Gemini calls and attack generation.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Service URLs
# ---------------------------------------------------------------------------
LOCAL_API_BASE = os.environ.get("SIGNPOST_API_BASE", "")
COMFY_SERVER = os.environ.get("SIGNPOST_COMFY_SERVER", "")
COMFY_SERVERS = os.environ.get("SIGNPOST_COMFY_SERVERS", "")  # Comma-separated for multi-GPU

# ---------------------------------------------------------------------------
# Default Model
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("SIGNPOST_DEFAULT_MODEL", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Evaluation Settings
# ---------------------------------------------------------------------------
DEFAULT_CONCURRENCY = int(os.environ.get("SIGNPOST_CONCURRENCY", "5"))
WLA_THRESHOLDS = [1, 25, 200, 750, 2500]  # km
TFR_RADIUS = 50  # km

# ---------------------------------------------------------------------------
# Probing / Defense Settings
# ---------------------------------------------------------------------------
PROBING_SUBSET_PER_DATASET = 250
GENERALIZATION_SAMPLE_SIZE = 200
HUMAN_VALIDATION_SAMPLE_SIZE = 120


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    return DATASETS[dataset_name]


def get_images_dir(dataset_name: str, variant: str) -> Path:
    """Get path to a specific image variant directory."""
    ds = get_dataset_config(dataset_name)
    if variant == "Original":
        return get_filtered_images_dir(dataset_name)

    return ds["images_dir"] / variant


def get_filtered_images_dir(dataset_name: str) -> Path:
    """Get path to filtered original images for one dataset."""
    get_dataset_config(dataset_name)  # Validate dataset name
    return DATA_ROOT / dataset_name / "filtered_images"


def get_dataset_results_dir(dataset_name: str, ensure_exists: bool = False) -> Path:
    """Get the standard results directory for one dataset."""
    get_dataset_config(dataset_name)  # Validate dataset name
    results_dir = RESULTS_DIR / dataset_name / "results"
    if ensure_exists:
        results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_model_results_dir(dataset_name: str, model_name: str, ensure_exists: bool = False) -> Path:
    """Get the results directory for one dataset/model combination."""
    model_results_dir = get_dataset_results_dir(dataset_name, ensure_exists=ensure_exists) / model_name
    if ensure_exists:
        model_results_dir.mkdir(parents=True, exist_ok=True)
    return model_results_dir


def get_output_path(dataset_name: str, variant: str, model_name: str) -> Path:
    """Get standard output JSONL path for a dataset/variant/model combination."""
    return get_model_results_dir(dataset_name, model_name, ensure_exists=True) / f"results_{variant}_{model_name}.jsonl"


def get_named_output_path(dataset_name: str, prefix: str, variant: str, model_name: str) -> Path:
    """Get a dataset-scoped output path for auxiliary experiment results.

    Results are organized into experiment-specific subdirectories:
      data/<dataset>/results/<model>/<prefix>/<prefix>_<Variant>_<Model>.jsonl
    e.g. data/im2gps3k/results/gpt-5.4/probing/probing_Adversarial_gpt-5.4.jsonl
    """
    model_dir = get_model_results_dir(dataset_name, model_name, ensure_exists=True)
    exp_dir = model_dir / prefix
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir / f"{prefix}_{variant}_{model_name}.jsonl"


def get_root_model_results_dir(model_name: str, ensure_exists: bool = False) -> Path:
    """Get a model-scoped directory under the repo-root results/ directory."""
    path = ROOT_RESULTS_DIR / model_name
    if ensure_exists:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_analysis_output_path(filename: str, ensure_parent: bool = False) -> Path:
    """Get a path under analysis/ for JSON or other analysis artifacts."""
    path = ANALYSIS_DIR / filename
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_analysis_plot_dir(dirname: str = "visualizations", ensure_exists: bool = False) -> Path:
    """Get a directory under analysis/ for generated plots."""
    path = ANALYSIS_DIR / dirname
    if ensure_exists:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_experiment_output_dir(
    experiment_name: str,
    model_name: str | None = None,
    ensure_exists: bool = False,
) -> Path:
    """Get a directory under analysis/experiment_outputs for experiment artifacts."""
    path = EXPERIMENT_OUTPUTS_DIR / experiment_name
    if model_name:
        path = path / model_name
    if ensure_exists:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_experiment_output_path(
    experiment_name: str,
    filename: str,
    model_name: str | None = None,
    ensure_parent: bool = False,
) -> Path:
    """Get a path under analysis/experiment_outputs for experiment outputs."""
    path = get_experiment_output_dir(experiment_name, model_name=model_name) / filename
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_subset_output_path(filename: str, ensure_parent: bool = False) -> Path:
    """Get a path under analysis/subsets for generated experiment subsets."""
    path = EXPERIMENT_SUBSETS_DIR / filename
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_paper_output_path(filename: str, ensure_parent: bool = False) -> Path:
    """Get a path under paper/ for generated paper assets."""
    path = PAPER_DIR / filename
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_api_key(provider: str) -> str:
    """Get API key for a provider.

    Raises:
        ValueError: If the requested provider's API key is not set.
    """
    key_map = {
        "sponsor": SPONSOR_API_KEY,
        "gemini": GEMINI_API_KEY,
    }
    if provider not in key_map:
        raise ValueError(f"Unknown provider '{provider}'. Available providers: {', '.join(sorted(key_map))}.")
    key = key_map.get(provider, "")
    if not key:
        env_var = f"{provider.upper()}_API_KEY"
        raise ValueError(
            f"API key for provider '{provider}' is not set. "
            f"Please set the {env_var} environment variable or add it to your .env file."
        )
    return key


def print_config():
    """Print current configuration for debugging."""
    print("=" * 50)
    print("  SIGNPOST-Bench Configuration")
    print("=" * 50)
    print(f"  CODE_DIR:     {CODE_DIR}")
    print(f"  DATA_ROOT:    {DATA_ROOT}")
    print(f"  RESULTS_ROOT: {RESULTS_DIR}")
    print(f"  DEFAULT_MODEL:{DEFAULT_MODEL}")
    print(f"  CONCURRENCY:  {DEFAULT_CONCURRENCY}")
    print(f"  Datasets:     {list(DATASETS.keys())}")
    for name, ds in DATASETS.items():
        exists = ds["images_dir"].exists()
        print(f"    {name}: {'[OK]' if exists else '[--]'} {ds['images_dir']}")
        print(f"      results -> {get_dataset_results_dir(name)}")
    print("  API Keys:")
    print(f"    Sponsor:     {'[OK]' if SPONSOR_API_KEY else '[--]'}")
    print(f"    Gemini:      {'[OK]' if GEMINI_API_KEY else '[--]'}")
    print(f"  Sponsor API Base: {SPONSOR_API_BASE or '[NOT SET]'}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
