#!/usr/bin/env python3
# generated through claude
"""
Script to randomly select images, shuffle their pixels, and add captions.
Usage: python shuffle_caption.py <directory_path> <number_of_images> <caption_text>
"""

import os
import sys
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def get_font(size=40):
    """Try to get a nice font, fall back to default if unavailable."""
    try:
        # Try to use a common system font
        return ImageFont.truetype("arial.ttf", size)
    except:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except:
            # Fall back to default font
            return ImageFont.load_default()

def shuffle_pixels(img):
    """Randomly shuffle all pixels in an image."""
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Get the shape
    height, width = img_array.shape[:2]
    
    # Reshape to 2D array of pixels (flatten spatial dimensions)
    if len(img_array.shape) == 3:  # Color image
        channels = img_array.shape[2]
        pixels = img_array.reshape(-1, channels)
    else:  # Grayscale
        pixels = img_array.reshape(-1)
    
    # Shuffle the pixels
    np.random.shuffle(pixels)
    
    # Reshape back to original image dimensions
    if len(img_array.shape) == 3:
        shuffled_array = pixels.reshape(height, width, channels)
    else:
        shuffled_array = pixels.reshape(height, width)
    
    # Convert back to PIL Image
    return Image.fromarray(shuffled_array.astype('uint8'))

def add_caption_to_image(img, caption_text,location="bottom"):
    """Add caption to an image at center-top."""
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Get font
    font_size = 40
    font = get_font(font_size)
    
    # Calculate text size and position
    bbox = draw.textbbox((0, 0), caption_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position text at center-top with some padding

    
    # Draw background rectangle for better readability
    # padding = 5
    padding = 8

    x = (img.width - text_width) // 2

    if location == "bottom":
        y = img.height - text_height - (padding * 3)
    else:
        y = 10

    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding * 2],
        fill='black'
    )
    
    # Draw the text
    draw.text((x, y), caption_text, fill='white', font=font)
    
    return img

def process_image(input_path, output_path, caption):
    """Shuffle pixels and add caption to an image."""
    try:
        # Open the image
        img = Image.open(input_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Shuffle pixels
        print(f"  Shuffling pixels...")
        shuffled_img = shuffle_pixels(img)
        
        # Add caption
        print(f"  Adding caption...")
        captioned_img = add_caption_to_image(shuffled_img, caption)
        
        # Save the image
        captioned_img.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_directory(dir_path, num_images, caption):
    """Randomly select n images, shuffle pixels, and add captions."""
    # Define supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # Get the directory path
    source_dir = Path(dir_path)
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Directory '{dir_path}' does not exist.")
        return
    
    if not source_dir.is_dir():
        print(f"Error: '{dir_path}' is not a directory.")
        return
    
    # Get all image files
    all_images = [f for f in source_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if len(all_images) == 0:
        print("Error: No images found in the directory.")
        return
    
    # Check if requested number is available
    if num_images > len(all_images):
        print(f"Warning: Only {len(all_images)} images available. Processing all of them.")
        num_images = len(all_images)
    
    # Randomly select n images
    selected_images = random.sample(all_images, num_images)
    print(f"Randomly selected {num_images} images from {len(all_images)} total images.")
    
    # Create new directory name
    parent_dir = source_dir.parent
    old_dir_name = source_dir.name
    new_dir_name = f"{old_dir_name}-shuffled"
    new_dir_path = parent_dir / new_dir_name
    
    # Create new directory
    try:
        new_dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {new_dir_path}\n")
    except Exception as e:
        print(f"Error creating directory: {e}")
        return
    
    # Process selected images
    processed_count = 0
    
    for idx, file_path in enumerate(selected_images, 1):
        print(f"[{idx}/{num_images}] Processing: {file_path.name}")
        output_path = new_dir_path / file_path.name
        
        if process_image(file_path, output_path, caption):
            processed_count += 1
        print()
    
    # Summary
    print(f"{'='*50}")
    print(f"Processing complete!")
    print(f"Images selected: {num_images}")
    print(f"Successfully processed: {processed_count}")
    print(f"Output directory: {new_dir_path}")
    print(f"{'='*50}")

def main():
    """Main function to parse arguments and run the script."""
    if len(sys.argv) != 4:
        print("Usage: python generate-text-concept.py <directory_path> <number_of_images> <caption_text>")
        print("Example: python generate-text-concept.py  ./my_images 10 'Shuffled Image'")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    
    try:
        num_images = int(sys.argv[2])
        if num_images <= 0:
            print("Error: Number of images must be a positive integer.")
            sys.exit(1)
    except ValueError:
        print("Error: Number of images must be a valid integer.")
        sys.exit(1)
    
    caption = sys.argv[3]
    
    process_directory(dir_path, num_images, caption)

if __name__ == "__main__":
    main()