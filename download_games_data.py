#!/usr/bin/env python3
"""
Download chess games data from HuggingFace repository
"""

import os
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def list_games_files(repo_name="jd0g/chess-gpt-eval", repo_type="dataset"):
    """List all files in the games/ directory of the repository"""
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found in .env file")
        return []
    
    try:
        print(f"📂 Listing files in {repo_name}/games/ (repo_type: {repo_type})...")
        files = list_repo_files(repo_id=repo_name, repo_type=repo_type, token=token)
        
        # Filter for games directory
        games_files = [f for f in files if f.startswith("games/")]
        
        print(f"Found {len(games_files)} files in games/ directory:")
        for file in sorted(games_files):
            print(f"  - {file}")
        
        return games_files
        
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []

def download_model_games(model_name, repo_name="jd0g/chess-gpt-eval", repo_type="dataset", output_dir="data"):
    """Download games data for a specific model"""
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found in .env file")
        return None
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # List files to find the right one for this model
        files = list_repo_files(repo_id=repo_name, repo_type=repo_type, token=token)
        
        # Look for files that contain the model name
        model_files = [f for f in files if f.startswith("games/") and model_name.lower() in f.lower()]
        
        if not model_files:
            print(f"❌ No games files found for model: {model_name}")
            return None
        
        print(f"📥 Found {len(model_files)} files for {model_name}:")
        for file in model_files:
            print(f"  - {file}")
        
        # Download each file
        downloaded_files = []
        for file_path in model_files:
            print(f"\n⬇️  Downloading {file_path}...")
            
            local_filename = f"{model_name}_{Path(file_path).name}"
            local_path = Path(output_dir) / local_filename
            
            downloaded_file = hf_hub_download(
                repo_id=repo_name,
                repo_type=repo_type,
                filename=file_path,
                token=token,
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Downloaded to: {downloaded_file}")
            downloaded_files.append(downloaded_file)
        
        return downloaded_files
        
    except Exception as e:
        print(f"❌ Error downloading files: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download chess games data from HuggingFace")
    parser.add_argument("--list", action="store_true", help="List available files")
    parser.add_argument("--model", type=str, help="Model name to download games for")
    parser.add_argument("--repo", default="jd0g/chess-gpt-eval", help="HuggingFace repository")
    parser.add_argument("--repo-type", default="dataset", help="Repository type (dataset or model)")
    parser.add_argument("--output", default="data", help="Output directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_games_files(args.repo, args.repo_type)
    elif args.model:
        download_model_games(args.model, args.repo, args.repo_type, args.output)
    else:
        print("Use --list to see available files or --model <name> to download specific model data")
 
 