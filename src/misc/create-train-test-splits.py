import os
import shutil
import random
import argparse
from tqdm import tqdm

def split_dataset(input_dir: str, split_ratio: float) -> None:
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not (0 < split_ratio < 1):
        raise ValueError(f"Split ratio must be between 0 and 1, got {split_ratio}")

    # Grab base directory name for naming output folders
    base_name = os.path.basename(os.path.normpath(input_dir))
    parent_dir = os.path.dirname(os.path.normpath(input_dir))

    split_1_dir = os.path.join(parent_dir, f"{base_name}-split-1")
    split_2_dir = os.path.join(parent_dir, f"{base_name}-split-2")

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if not files:
        raise ValueError("No files found in the input directory.")

    random.shuffle(files)

    split_index = int(len(files) * split_ratio)
    split_1_files = files[:split_index]
    split_2_files = files[split_index:]

    os.makedirs(split_1_dir, exist_ok=True)
    os.makedirs(split_2_dir, exist_ok=True)

    for f in tqdm(split_1_files,desc="Split 1 ... "):
        shutil.copy2(os.path.join(input_dir, f), os.path.join(split_1_dir, f))
    for f in tqdm(split_2_files,desc="Split 2 ... "):
        shutil.copy2(os.path.join(input_dir, f), os.path.join(split_2_dir, f))

    print(
        f"Split complete.\n"
        f"{len(split_1_files)} files → {split_1_dir}\n"
        f"{len(split_2_files)} files → {split_2_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a directory of files into two random sets.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory.")
    parser.add_argument("--split_ratio", type=float, required=True, help="Split ratio (0-1).")
    args = parser.parse_args()

    split_dataset(args.input_dir, args.split_ratio)
