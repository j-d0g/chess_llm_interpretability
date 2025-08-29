#!/usr/bin/env python3
"""
Upload chess piece probes to HuggingFace Hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, CommitOperationAdd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Upload chess piece probes to HuggingFace')
    parser.add_argument('--username', required=True, help='HuggingFace username')
    parser.add_argument('--repo_name', required=True, help='Repository name')
    parser.add_argument('--token', help='HuggingFace token (or set HF_TOKEN env var)')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be uploaded without uploading')
    parser.add_argument('--resume', action='store_true', help='Skip files already present in the HF repo')
    parser.add_argument('--bulk', action='store_true', help='Upload remaining files in a single bulk commit (reduces rate limits)')
    parser.add_argument('--throttle', type=float, default=0.0, help='Sleep seconds between uploads when not using bulk')
    parser.add_argument('--max_retries', type=int, default=8, help='Max retries per file on transient errors')
    parser.add_argument('--initial_backoff', type=float, default=60.0, help='Initial backoff seconds on 429 rate limits')
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.getenv('HF_TOKEN')
    if not token and not args.dry_run:
        print("Error: HuggingFace token required. Set HF_TOKEN env var or use --token")
        sys.exit(1)
    
    # Initialize API
    if not args.dry_run:
        api = HfApi(token=token)
        repo_id = f"{args.username}/{args.repo_name}"
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_id, token=token, exist_ok=True)
            print(f"✅ Repository {repo_id} ready")
        except Exception as e:
            print(f"❌ Error creating repository: {e}")
            sys.exit(1)
    
    # Find all probe files
    probe_dir = Path("linear_probes")
    probe_files = list(probe_dir.glob("*chess_piece_probe*.pth"))
    
    print(f"Found {len(probe_files)} probe files to upload")
    
    # Organize files by model
    trained_files = [f for f in probe_files if "RANDOM" not in f.name]
    random_files = [f for f in probe_files if "RANDOM" in f.name]
    
    print(f"- Trained model probes: {len(trained_files)}")
    print(f"- Random model probes: {len(random_files)}")

    # If resuming, filter out files already present on HF
    if args.resume and not args.dry_run:
        try:
            remote_files = set(HfApi(token=token).list_repo_files(repo_id=repo_id))
        except Exception as e:
            print(f"❌ Error listing repo files: {e}")
            sys.exit(1)
        before_count = len(probe_files)
        probe_files = [f for f in probe_files if f.name not in remote_files]
        print(f"🔁 Resume mode: {before_count - len(probe_files)} already on HF; {len(probe_files)} remaining")
    
    if args.dry_run:
        print("\n=== DRY RUN - Files that would be uploaded ===")
        for probe_file in sorted(probe_files):
            print(f"  {probe_file}")
        return
    
    # Upload files
    print(f"\n🚀 Starting upload to {repo_id}...")

    uploaded_count = 0
    failed_count = 0

    if args.bulk and not args.dry_run:
        try:
            operations = [
                CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f))
                for f in probe_files
            ]
            if not operations:
                print("Nothing to upload (already up to date).")
            else:
                api.create_commit(
                    repo_id=repo_id,
                    operations=operations,
                    commit_message=f"Add {len(operations)} probe files"
                )
                uploaded_count = len(operations)
                for idx, f in enumerate(probe_files, start=1):
                    print(f"✅ Uploaded {f.name} ({idx}/{len(probe_files)})")
        except Exception as e:
            failed_count = len(probe_files)
            print(f"❌ Bulk upload failed: {e}")
    else:
        import time
        for probe_file in probe_files:
            attempts = 0
            while True:
                try:
                    api.upload_file(
                        path_or_fileobj=str(probe_file),
                        path_in_repo=probe_file.name,
                        repo_id=repo_id,
                        commit_message=f"Add {probe_file.name}"
                    )
                    uploaded_count += 1
                    print(f"✅ Uploaded {probe_file.name} ({uploaded_count}/{len(probe_files)})")
                    if args.throttle > 0:
                        time.sleep(args.throttle)
                    break
                except Exception as e:
                    msg = str(e)
                    rate_limited = ('Too Many Requests' in msg) or ('429' in msg)
                    attempts += 1
                    if rate_limited and attempts <= args.max_retries:
                        sleep_s = min(args.initial_backoff * (2 ** (attempts - 1)), 900)
                        print(f"⏳ Rate limited uploading {probe_file.name} (attempt {attempts}/{args.max_retries}). Sleeping {int(sleep_s)}s then retrying...")
                        time.sleep(sleep_s)
                        continue
                    failed_count += 1
                    print(f"❌ Failed to upload {probe_file.name}: {e}")
                    break
    
    print(f"\n📊 Upload Summary:")
    print(f"  ✅ Successfully uploaded: {uploaded_count}")
    print(f"  ❌ Failed uploads: {failed_count}")
    print(f"  📁 Total files: {len(probe_files)}")
    
    if uploaded_count > 0:
        print(f"\n🎉 Probes uploaded to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
