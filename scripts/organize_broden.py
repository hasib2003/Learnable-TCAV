import os
import shutil
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

# Increase max pixel limit for large images if necessary
Image.MAX_IMAGE_PIXELS = None

def extract_broden_concept(dataset_root, concept_name, output_dir):
    
    label_file = os.path.join(dataset_root, 'label.csv')
    index_file = os.path.join(dataset_root, 'index.csv')
    image_folder = os.path.join(dataset_root, 'images')

    # 1. Load Metadata
    if not os.path.exists(label_file) or not os.path.exists(index_file):
        print(f"Error: label.csv or index.csv not found in {dataset_root}")
        sys.exit(1)

    print(f"Loading metadata...")
    df_labels = pd.read_csv(label_file)
    df_index = pd.read_csv(index_file)

    # 2. Find Concept Details
    # Filter by name case-insensitive
    match = df_labels[df_labels['name'].str.lower() == concept_name.lower()]

    if match.empty:
        print(f"Error: Concept '{concept_name}' not found in label.csv.")
        sys.exit(1)

    # Get ID and Raw Category
    target_id = match.iloc[0]['number']
    raw_category = match.iloc[0]['category'] 
    
    # FIX 1: Clean the category name. 
    # "material(9905)" -> "material"
    # We split by '(' and take the first part.
    target_category = raw_category.split('(')[0].strip()

    print(f"--- Target Found ---")
    print(f"Concept:  {concept_name}")
    print(f"ID:       {target_id}")
    print(f"Category: {target_category} (derived from '{raw_category}')")
    print(f"--------------------")

    # 3. Validate Column in Index
    # We strip whitespace from columns just in case
    df_index.columns = [c.strip() for c in df_index.columns]
    
    if target_category not in df_index.columns:
        print(f"Error: Cleaned category '{target_category}' is not a column in index.csv.")
        print(f"Available columns: {list(df_index.columns)}")
        sys.exit(1)

    # 4. Filter Logic
    print(f"Scanning dataset for '{concept_name}' (ID: {target_id})...")
    print("Note: Since index.csv points to masks, we must open files to check for the concept. This might take a moment.")

    os.makedirs(output_dir, exist_ok=True)
    
    found_count = 0
    missing_files = 0

    # Iterate over all rows in the index
    for idx, row in tqdm(df_index.iterrows(), total=len(df_index)):
        
        # Get the value in the category column (e.g., path to material mask)
        mask_val = row[target_category]
        
        # If the value is NaN or empty, skip
        if pd.isna(mask_val) or str(mask_val).strip() == '':
            continue

        is_present = False
        
        # Logic A: If the column contains a Path (string) -> Open Image Mask
        if isinstance(mask_val, str) and not mask_val.isdigit():
            mask_path = os.path.join(image_folder, mask_val.strip())
            
            if not os.path.exists(mask_path):
                # Try checking without 'images' folder prefix if implied
                mask_path = os.path.join(dataset_root, mask_val.strip())
            
            if os.path.exists(mask_path):
                try:
                    # Open mask, convert to numpy array
                    # Broden masks are usually single channel (L) or palette (P)
                    mask_img = Image.open(mask_path)
                    mask_arr = np.array(mask_img)
                    
                    # Check if the target_id exists in the mask pixels
                    if target_id in mask_arr:
                        is_present = True
                except Exception as e:
                    # If corrupt image, skip
                    pass
            else:
                missing_files += 1

        # Logic B: If the column contains direct Numbers (like 'scene' or 'class')
        else:
            try:
                # Handle cases like "12;14" or just "83"
                val_str = str(mask_val)
                ids_in_row = [int(x) for x in val_str.replace(';', ' ').split() if x.isdigit()]
                if target_id in ids_in_row:
                    is_present = True
            except:
                pass

        # 5. Copy Image if Concept Found
        if is_present:
            # The main image path is in the 'image' column
            rel_img_path = row['image']
            src_img_path = os.path.join(image_folder, rel_img_path)
            
            # Use just the filename for destination
            dst_filename = os.path.basename(rel_img_path)
            dst_img_path = os.path.join(output_dir, dst_filename)
            
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
                found_count += 1
            else:
                # Try finding it in root
                src_img_path_alt = os.path.join(dataset_root, rel_img_path)
                if os.path.exists(src_img_path_alt):
                    shutil.copy2(src_img_path_alt, dst_img_path)
                    found_count += 1

    print(f"\nExtraction complete.")
    print(f"Images extracted: {found_count}")
    if missing_files > 0:
        print(f"Warning: Could not locate {missing_files} mask files referenced in index.csv.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--concept', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    extract_broden_concept(args.dataset, args.concept, args.output)