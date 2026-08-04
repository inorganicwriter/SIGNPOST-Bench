import argparse
import logging
import shutil
from pathlib import Path

import easyocr
from tqdm import tqdm

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Filter images containing text using EasyOCR")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing raw images")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save images with text")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Confidence threshold for text detection")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize EasyOCR
    logging.info(f"Initializing EasyOCR (GPU={args.gpu})...")
    reader = easyocr.Reader(["en"], gpu=args.gpu)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    logging.info(f"Found {len(images)} images. Process started...")

    kept_count = 0

    for img_path in tqdm(images, desc="Filtering Images"):
        try:
            # Read text
            # detail=0 returns just list of text strings
            results = reader.readtext(str(img_path), detail=0, paragraph=True)

            # Simple heuristic: If any text found, keep it
            # You can add more complex logic here (e.g. min length)
            has_text = False
            for text in results:
                if len(text.strip()) > 1:  # Ignore single chars or empty
                    has_text = True
                    break

            if has_text:
                shutil.copy2(img_path, output_dir / img_path.name)
                kept_count += 1

        except Exception as e:
            logging.error(f"Error processing {img_path.name}: {e}")
            continue

    logging.info("Filtering complete.")
    logging.info(f"Total processed: {len(images)}")
    logging.info(f"Kept (Found Text): {kept_count}")
    logging.info(f"Discarded: {len(images) - kept_count}")
    logging.info(f"Filtered images saved to: {output_dir}")


if __name__ == "__main__":
    main()
