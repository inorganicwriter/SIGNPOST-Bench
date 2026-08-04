import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Convert User CSV Metadata to SIGNPOST-Bench Formats")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path (e.g. yfcc4k_metadata_address.csv)")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for converted metadata")
    parser.add_argument("--dataset-name", type=str, default="yfcc4k", help="Dataset name prefix")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Files to generate
    # 1. clean_metadata.jsonl (For generate_attacks.py)
    #    Format: {"filename": "...", "output_filename": "...", "prompt": "..."}
    # 2. gt_dataset.txt (For evaluate.py)
    #    Format: Old YFCC100m format (TSV).
    #    Required columns expected by evaluate.py:
    #      Index 1: ID/Name
    #      Index 10: Longitude
    #      Index 11: Latitude
    #      Index 14: URL (path ending with filename)

    jsonl_path = os.path.join(args.out_dir, f"{args.dataset_name}_clean_meta.jsonl")
    tsv_path = os.path.join(args.out_dir, f"{args.dataset_name}_gt.tsv")

    print(f"Converting {args.csv}...")

    count = 0
    with open(args.csv, encoding="utf-8", errors="ignore") as f:
        # Assuming typical CSV with headers, or user can adjust
        # If user csv has headers: filename,lat,lon,...
        # Let's try to sniff or assume standard structure based on user description
        # If we can't be sure, we write a generic reader assuming typical columns
        reader = csv.DictReader(f)

        with open(jsonl_path, "w", encoding="utf-8") as f_json, open(tsv_path, "w", encoding="utf-8") as f_tsv:
            for row in reader:
                # Try multiple common column names for compatibility across datasets
                # yfcc4k: id, latitude, longitude
                # im2gps3k: photo_id, latitude_x, longitude_x
                filename = row.get("filename") or row.get("Image_ID") or row.get("photo_id") or row.get("id")
                lat = row.get("latitude") or row.get("lat") or row.get("latitude_x") or row.get("latitude_y")
                lon = (
                    row.get("longitude")
                    or row.get("lon")
                    or row.get("lng")
                    or row.get("longitude_x")
                    or row.get("longitude_y")
                )

                if not filename or not lat or not lon:
                    continue

                # Extract geographic info for geo-aware attack generation
                city = (row.get("city") or row.get("City") or row.get("locality") or "").strip()
                county = (row.get("county") or row.get("County") or row.get("district") or "").strip()
                province = (
                    row.get("province")
                    or row.get("state")
                    or row.get("Province")
                    or row.get("State")
                    or row.get("admin1")
                    or ""
                ).strip()
                country = (row.get("country") or row.get("Country") or row.get("country_code") or "").strip()

                # 1. JSONL Entry (includes geo info for generate_attacks.py)
                entry = {
                    "filename": filename,
                    "output_filename": f"Clean/{filename}",
                    "id": filename,
                    "city": city,
                    "county": county,
                    "province": province,
                    "country": country,
                }
                f_json.write(json.dumps(entry) + "\n")

                # 2. TSV Entry (Mocking YFCC format)
                # evaluate.py needs:
                # parts[1] (ID)
                # parts[10] (Lon)
                # parts[11] (Lat)
                # parts[14] (URL/Path)

                dummy = ["0"] * 20
                dummy[1] = filename
                dummy[10] = str(lon)
                dummy[11] = str(lat)
                dummy[14] = f"http://dummy/{filename}"

                f_tsv.write("\t".join(dummy) + "\n")
                count += 1

    print(f"Converted {count} entries.")
    print(f"Saved to:\n  - {jsonl_path}\n  - {tsv_path}")
    print("NOTE: Please check the CSV column names in the script if 0 entries were converted.")


if __name__ == "__main__":
    main()
