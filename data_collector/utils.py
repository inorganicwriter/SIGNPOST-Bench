import json
import os


def save_metadata(output_folder, entry):
    """Append metadata entry to metadata.jsonl in the output folder."""
    meta_path = os.path.join(output_folder, "metadata.jsonl")
    try:
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Warn] Failed to save metadata: {e}")


def load_workflow_api(workflow_file):
    """Load the ComfyUI API JSON workflow file."""
    try:
        with open(workflow_file, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Workflow file {workflow_file} not found.")
        return None
