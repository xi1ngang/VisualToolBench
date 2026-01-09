"""
Download VisualToolBench dataset from HuggingFace and extract to local files.
"""
from datasets import load_dataset
import json
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")

# Load dataset directly from Hugging Face
print("Downloading dataset from HuggingFace...")
ds = load_dataset("ScaleAI/VisualToolBench")

# Create output directories
single_turn_dir = os.path.join(OUTPUT_DIR, "single_turn")
multi_turn_dir = os.path.join(OUTPUT_DIR, "multi_turn")
os.makedirs(os.path.join(single_turn_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(multi_turn_dir, "images"), exist_ok=True)

# Process each sample
single_turn_samples = []
multi_turn_samples = []
single_idx = 0
multi_idx = 0

for idx, sample in enumerate(ds['test']):
    # Determine if single-turn or multi-turn based on number of prompts
    is_single_turn = len(sample['turn_prompts']) == 1
    
    if is_single_turn:
        base_dir = single_turn_dir
        sample_idx = single_idx
        single_idx += 1
    else:
        base_dir = multi_turn_dir
        sample_idx = multi_idx
        multi_idx += 1
    
    sample_data = {}
    
    # Copy all non-image fields
    for key in sample.keys():
        if key not in ['images', 'images_by_turn']:
            sample_data[key] = sample[key]
    
    # Save images
    sample_images_dir = os.path.join(base_dir, "images", f"sample_{sample_idx}")
    os.makedirs(sample_images_dir, exist_ok=True)
    
    # Save main images list with RELATIVE paths
    image_paths = []
    if sample['images']:
        for img_idx, img in enumerate(sample['images']):
            img_filename = f"image_{img_idx}.png"
            img_full_path = os.path.join(sample_images_dir, img_filename)
            img.save(img_full_path)
            # Store relative path from data directory
            if is_single_turn:
                rel_path = f"data/single_turn/images/sample_{sample_idx}/{img_filename}"
            else:
                rel_path = f"data/multi_turn/images/sample_{sample_idx}/{img_filename}"
            image_paths.append(rel_path)
    sample_data['image_paths'] = image_paths
    
    # Save images by turn with RELATIVE paths
    images_by_turn_paths = []
    if sample['images_by_turn']:
        for turn_idx, turn_images in enumerate(sample['images_by_turn']):
            turn_paths = []
            if turn_images:
                for img_idx, img in enumerate(turn_images):
                    img_filename = f"turn_{turn_idx}_image_{img_idx}.png"
                    img_full_path = os.path.join(sample_images_dir, img_filename)
                    img.save(img_full_path)
                    # Store relative path from data directory
                    if is_single_turn:
                        rel_path = f"data/single_turn/images/sample_{sample_idx}/{img_filename}"
                    else:
                        rel_path = f"data/multi_turn/images/sample_{sample_idx}/{img_filename}"
                    turn_paths.append(rel_path)
            images_by_turn_paths.append(turn_paths)
    sample_data['images_by_turn_paths'] = images_by_turn_paths
    
    if is_single_turn:
        single_turn_samples.append(sample_data)
    else:
        multi_turn_samples.append(sample_data)
    
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1} samples...")

# Save metadata as JSON for each category
single_json_path = os.path.join(single_turn_dir, "dataset.json")
with open(single_json_path, 'w') as f:
    json.dump(single_turn_samples, f, indent=2)

multi_json_path = os.path.join(multi_turn_dir, "dataset.json")
with open(multi_json_path, 'w') as f:
    json.dump(multi_turn_samples, f, indent=2)

print(f"\nDone!")
print(f"  Single-turn: {len(single_turn_samples)} samples -> {single_turn_dir}")
print(f"  Multi-turn:  {len(multi_turn_samples)} samples -> {multi_turn_dir}")
