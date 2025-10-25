import os
import sys
import random
import shutil
from tqdm import tqdm

def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py <src_dir> <dst_dir> <n_samples> <seed>")
        sys.exit(1)

    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    n_samples = int(sys.argv[3])
    seed = int(sys.argv[4])

    # Validate source directory
    if not os.path.isdir(src_dir):
        print(f"Error: Source directory '{src_dir}' does not exist.")
        sys.exit(1)

    # Destination subdir = dst_dir/random_<seed>
    target_dir = os.path.join(dst_dir, f"random_{seed}")
    os.makedirs(target_dir, exist_ok=True)

    # List files in source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    if not files:
        print("Error: No files found in source directory.")
        sys.exit(1)

    if n_samples > len(files):
        print(f"Warning: Requested {n_samples} samples, but only {len(files)} files available.")
        n_samples = len(files)

    random.seed(seed)
    sampled_files = random.sample(files, n_samples)

    # Copy sampled files with original names
    for f in tqdm(sampled_files,desc="copying files ... "):
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(target_dir, f)
        shutil.copy2(src_path, dst_path)

    print(f"Copied {len(sampled_files)} files to {target_dir}")

if __name__ == "__main__":
    main()
