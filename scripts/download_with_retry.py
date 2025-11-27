#!/usr/bin/env python3
"""
Download Wiki-18 corpus and E5 index with retry mechanism
"""
import argparse
from huggingface_hub import hf_hub_download
import time

parser = argparse.ArgumentParser(description="Download files from Hugging Face with retry.")
parser.add_argument("--repo_id", type=str, default="PeterJinGo/wiki-18-e5-index", help="Hugging Face repository ID")
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries")

args = parser.parse_args()

def download_with_retry(repo_id, filename, local_dir, max_retries=5):
    """Download a file with retry mechanism"""
    for attempt in range(max_retries):
        try:
            print(f"\n{'='*70}")
            print(f"Downloading {filename} (Attempt {attempt + 1}/{max_retries})")
            print(f"Repository: {repo_id}")
            print(f"{'='*70}\n")

            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=local_dir,
                resume_download=True,  # Enable resume
            )

            print(f"\n✓ Successfully downloaded {filename}\n")
            return True

        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"Retrying in {wait_time} seconds...\n")
                time.sleep(wait_time)
            else:
                print(f"\n✗ Failed to download {filename} after {max_retries} attempts")
                return False

    return False

# Download E5 index files (split into 2 parts)
print("\n" + "="*70)
print("Downloading E5 Index Files (Wiki-18)")
print("="*70)

repo_id = "PeterJinGo/wiki-18-e5-index"
files = ["part_aa", "part_ab"]

for file in files:
    success = download_with_retry(repo_id, file, args.save_path, args.max_retries)
    if not success:
        print(f"\n✗ Critical error: Failed to download {file}")
        print("Please check your network connection and try again.")
        exit(1)

# Download corpus
print("\n" + "="*70)
print("Downloading Wiki-18 Corpus")
print("="*70)

repo_id = "PeterJinGo/wiki-18-corpus"
success = download_with_retry(repo_id, "wiki-18.jsonl.gz", args.save_path, args.max_retries)

if not success:
    print("\n✗ Critical error: Failed to download corpus")
    print("Please check your network connection and try again.")
    exit(1)

print("\n" + "="*70)
print("Download Complete!")
print("="*70)
print("\nNext steps:")
print(f"1. cd {args.save_path}")
print("2. cat part_aa part_ab > e5_Flat.index")
print("3. gzip -d wiki-18.jsonl.gz")
print("="*70 + "\n")
