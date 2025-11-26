#!/usr/bin/env python3
"""
Download QReCC dataset from HuggingFace and save to local data directory.

Usage:
    python scripts/download_qrecc_dataset.py --output_dir data/qrecc_raw
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_qrecc_dataset(output_dir: str):
    """
    Download QReCC dataset from HuggingFace Hub and save as JSONL files.

    Args:
        output_dir: Directory to save the downloaded dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading QReCC dataset from HuggingFace...")
    print("This may take a few minutes depending on your network speed.")

    # Load QReCC dataset from HuggingFace
    # The dataset name might be: "BeIR/qrecc" or directly accessible
    try:
        # Try loading from BeIR collection first
        dataset = load_dataset("BeIR/qrecc", "queries")
        print("Successfully loaded QReCC dataset from BeIR/qrecc")
    except Exception as e:
        print(f"Failed to load from BeIR/qrecc: {e}")
        print("Attempting alternative loading method...")

        # Alternative: Load from alternative sources or manually
        # You may need to adjust this based on the actual HuggingFace dataset path
        try:
            dataset = load_dataset("qrecc")
            print("Successfully loaded QReCC dataset")
        except Exception as e2:
            print(f"Error: {e2}")
            print("\nPlease manually download QReCC dataset from:")
            print("https://github.com/apple/ml-qrecc")
            print("Or check HuggingFace datasets: https://huggingface.co/datasets")
            return

    # Save each split as JSONL
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        output_file = output_path / f"{split_name}.jsonl"

        print(f"\nSaving {split_name} split to {output_file}...")
        print(f"Total examples: {len(split_data)}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(split_data, desc=f"Writing {split_name}"):
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')

        print(f"✓ Saved {split_name} split with {len(split_data)} examples")

    print(f"\n✓ Dataset download complete! Files saved to: {output_path}")
    print("\nDataset structure:")
    for file in sorted(output_path.glob("*.jsonl")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.2f} MB)")


def download_qrecc_from_github(output_dir: str):
    """
    Alternative method: Download QReCC dataset from official GitHub repo.

    Args:
        output_dir: Directory to save the downloaded dataset
    """
    import urllib.request
    import zipfile

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Official QReCC GitHub URLs
    urls = {
        "train": "https://raw.githubusercontent.com/apple/ml-qrecc/main/dataset/qrecc_train.json",
        "test": "https://raw.githubusercontent.com/apple/ml-qrecc/main/dataset/qrecc_test.json",
    }

    print("Downloading QReCC dataset from official GitHub repository...")

    for split_name, url in urls.items():
        output_file = output_path / f"{split_name}.json"
        print(f"\nDownloading {split_name} split from {url}...")

        try:
            urllib.request.urlretrieve(url, output_file)
            print(f"✓ Downloaded {split_name} split to {output_file}")

            # Convert to JSONL format
            jsonl_file = output_path / f"{split_name}.jsonl"
            with open(output_file, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)

            with open(jsonl_file, 'w', encoding='utf-8') as f_out:
                for item in data:
                    json.dump(item, f_out, ensure_ascii=False)
                    f_out.write('\n')

            print(f"✓ Converted to JSONL format: {jsonl_file}")

        except Exception as e:
            print(f"✗ Failed to download {split_name}: {e}")

    print(f"\n✓ Download complete! Files saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download QReCC dataset from HuggingFace")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/qrecc_raw",
        help="Directory to save the downloaded dataset (default: data/qrecc_raw)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "github"],
        help="Download source: huggingface or github (default: huggingface)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("QReCC Dataset Downloader")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print(f"Download source: {args.source}")
    print("=" * 70)

    if args.source == "huggingface":
        download_qrecc_dataset(args.output_dir)
    else:
        download_qrecc_from_github(args.output_dir)

    print("\n" + "=" * 70)
    print("Next steps:")
    print("=" * 70)
    print("1. Process the dataset:")
    print(f"   python scripts/data_process/qrecc_search_plan_b.py \\")
    print(f"       --input_dir {args.output_dir} \\")
    print(f"       --output_dir data/qrecc_plan_b")
    print("\n2. Start training:")
    print("   bash configs/qrecc/train_qrecc_ppo_plan_b.sh")
    print("=" * 70)


if __name__ == "__main__":
    main()