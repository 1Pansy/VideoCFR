import os
import zipfile
import argparse
from pathlib import Path

def extract_zip_files(root_dir):
    """
    Traverse the specified directory and all its subdirectories,
    and extract all zip files.
    Each zip file will be extracted into its containing directory.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.zip'):
                zip_path = os.path.join(dirpath, filename)
                print(f"Extracting: {zip_path}")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(dirpath)
                    print(f"Successfully extracted: {zip_path}")
                except Exception as e:
                    print(f"Failed to extract {zip_path}: {e}")

if __name__ == '__main__':
    default_root = Path(__file__).resolve().parent / "r1-v" / "Video-R1-data"
    parser = argparse.ArgumentParser(description="Extract zip files under a data directory.")
    parser.add_argument("--root", default=str(default_root), help="Root directory to scan for zip files.")
    args = parser.parse_args()
    extract_zip_files(args.root)
