"""
Create Preprocessed Dataset for MobileNetV2
Converts all images to: 224x224, RGB (3 channels), optimized for training
"""

import os
from PIL import Image
from tqdm import tqdm
import shutil
from pathlib import Path

# ==================== CONFIGURATION ====================

# INPUT: Your current dataset
INPUT_DATASET = r"C:\edge-ai-defect-classification\dataset_128"

# OUTPUT: New preprocessed dataset
OUTPUT_DATASET = r"C:\edge-ai-defect-classification\dataset_224_rgb"

# Target specifications (MobileNetV2 requirements)
TARGET_SIZE = 224  # 224x224 for MobileNetV2
TARGET_MODE = 'RGB'  # 3 channels
QUALITY = 95  # JPEG quality (95 = high quality, minimal artifacts)

# ==================== FUNCTIONS ====================

def count_images(root_dir):
    """Count total images in dataset"""
    count = 0
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(root_dir, split)
        if os.path.exists(split_dir):
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.isdir(class_dir):
                    count += len([f for f in os.listdir(class_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    return count


def preprocess_image(input_path, output_path, target_size=224, target_mode='RGB', quality=95):
    """
    Preprocess a single image
    
    Args:
        input_path: Path to input image
        output_path: Path to save preprocessed image
        target_size: Target dimension (224 for MobileNetV2)
        target_mode: Target color mode ('RGB' for 3 channels)
        quality: JPEG quality (1-100)
    
    Returns:
        bool: Success status
    """
    
    try:
        # Open image
        img = Image.open(input_path)
        
        # Convert to target mode (RGB = 3 channels)
        if img.mode != target_mode:
            if img.mode == 'L':
                # Grayscale → RGB: Duplicate channel 3 times
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                # RGBA → RGB: Remove alpha channel
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha as mask
                img = background
            elif img.mode == 'P':
                # Palette → RGB
                img = img.convert('RGB')
            else:
                # Any other mode → RGB
                img = img.convert('RGB')
        
        # Resize to target size
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Save as JPEG with high quality
        output_filename = os.path.splitext(os.path.basename(output_path))[0] + '.jpg'
        output_path_jpg = os.path.join(os.path.dirname(output_path), output_filename)
        
        img.save(output_path_jpg, 'JPEG', quality=quality, optimize=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error processing {input_path}: {e}")
        return False


def create_preprocessed_dataset(input_root, output_root, target_size=224, target_mode='RGB', quality=95):
    """
    Create complete preprocessed dataset
    
    Args:
        input_root: Root directory of original dataset
        output_root: Root directory for preprocessed dataset
        target_size: Target image dimension
        target_mode: Target color mode
        quality: JPEG quality
    """
    
    print("\n" + "="*80)
    print("🔄 DATASET PREPROCESSING - MOBILENETV2 OPTIMIZATION")
    print("="*80)
    
    print(f"\n📂 Input:  {input_root}")
    print(f"📂 Output: {output_root}")
    print(f"\n⚙️  Settings:")
    print(f"   Target size: {target_size}x{target_size}")
    print(f"   Color mode:  {target_mode} (3 channels)")
    print(f"   Quality:     {quality}%")
    
    # Count total images
    total_images = count_images(input_root)
    print(f"\n📊 Total images to process: {total_images}")
    
    # Create output directory
    if os.path.exists(output_root):
        print(f"\n⚠️  Output directory exists: {output_root}")
        response = input("   Overwrite? (yes/no): ").strip().lower()
        if response != 'yes':
            print("\n❌ Cancelled by user")
            return
        shutil.rmtree(output_root)
    
    os.makedirs(output_root, exist_ok=True)
    
    # Process each split (train, val, test)
    splits = ['train', 'val', 'test']
    
    total_processed = 0
    total_failed = 0
    
    for split in splits:
        split_input = os.path.join(input_root, split)
        split_output = os.path.join(output_root, split)
        
        if not os.path.exists(split_input):
            print(f"\n⚠️  Skipping {split}/ (not found)")
            continue
        
        print(f"\n{'='*80}")
        print(f"📁 Processing {split.upper()}/ split")
        print(f"{'='*80}")
        
        # Get all class folders
        classes = sorted([d for d in os.listdir(split_input) 
                         if os.path.isdir(os.path.join(split_input, d))])
        
        split_total = 0
        split_failed = 0
        
        for class_name in classes:
            input_class_dir = os.path.join(split_input, class_name)
            output_class_dir = os.path.join(split_output, class_name)
            
            # Create output class directory
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Get all images
            image_files = [f for f in os.listdir(input_class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
            
            print(f"\n  📂 {class_name}/ ({len(image_files)} images)")
            
            # Process each image
            for img_file in tqdm(image_files, desc=f"    Converting", ncols=80):
                input_path = os.path.join(input_class_dir, img_file)
                output_path = os.path.join(output_class_dir, img_file)
                
                success = preprocess_image(input_path, output_path, target_size, target_mode, quality)
                
                if success:
                    split_total += 1
                    total_processed += 1
                else:
                    split_failed += 1
                    total_failed += 1
        
        print(f"\n  ✅ {split.upper()} complete: {split_total} processed, {split_failed} failed")
    
    # Calculate sizes
    print("\n" + "="*80)
    print("📊 CALCULATING DISK SPACE SAVINGS")
    print("="*80)
    
    def get_dir_size(path):
        """Calculate directory size in MB"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total / (1024 * 1024)
    
    original_size = get_dir_size(input_root)
    new_size = get_dir_size(output_root)
    
    print(f"\n💾 Storage Analysis:")
    print(f"   Original dataset:  {original_size:.2f} MB")
    print(f"   Processed dataset: {new_size:.2f} MB")
    
    if new_size < original_size:
        savings = original_size - new_size
        savings_pct = (savings / original_size) * 100
        print(f"   Space saved:       {savings:.2f} MB ({savings_pct:.1f}%)")
    else:
        increase = new_size - original_size
        increase_pct = (increase / original_size) * 100
        print(f"   Size increase:     {increase:.2f} MB ({increase_pct:.1f}%)")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   Total processed:   {total_processed} images")
    print(f"   Failed:            {total_failed} images")
    print(f"   Success rate:      {(total_processed/(total_processed+total_failed)*100):.1f}%")
    
    print(f"\n📂 Preprocessed dataset saved to:")
    print(f"   {output_root}")
    
    print(f"\n⚙️  Dataset specifications:")
    print(f"   ✅ Size: {target_size}x{target_size}")
    print(f"   ✅ Mode: {target_mode} (3 channels)")
    print(f"   ✅ Format: JPEG (quality {quality}%)")
    print(f"   ✅ Ready for MobileNetV2 training!")
    
    print(f"\n🔧 NEXT STEP:")
    print(f"   Update config.py:")
    print(f"   DATASET_ROOT = r\"{output_root}\"")
    
    return output_root


def verify_preprocessed_dataset(dataset_root, expected_size=224, expected_mode='RGB'):
    """
    Verify all images in preprocessed dataset meet requirements
    
    Args:
        dataset_root: Root directory of preprocessed dataset
        expected_size: Expected image dimension
        expected_mode: Expected color mode
    """
    
    print("\n" + "="*80)
    print("🔍 VERIFYING PREPROCESSED DATASET")
    print("="*80)
    
    issues = []
    checked = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            continue
        
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_file in os.listdir(class_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    img = Image.open(img_path)
                    checked += 1
                    
                    # Check size
                    if img.size != (expected_size, expected_size):
                        issues.append(f"Wrong size: {img_path} ({img.size})")
                    
                    # Check mode
                    if img.mode != expected_mode:
                        issues.append(f"Wrong mode: {img_path} ({img.mode})")
                    
                except Exception as e:
                    issues.append(f"Cannot open: {img_path} ({e})")
    
    print(f"\n✅ Checked {checked} images")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues)-10} more")
    else:
        print(f"\n✅ All images verified successfully!")
        print(f"   ✅ Size: {expected_size}x{expected_size}")
        print(f"   ✅ Mode: {expected_mode}")
    
    return len(issues) == 0


def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("🚀 MOBILENETV2 DATASET PREPROCESSOR")
    print("   Convert any dataset to MobileNetV2-ready format")
    print("="*80)
    
    # Check input exists
    if not os.path.exists(INPUT_DATASET):
        print(f"\n❌ ERROR: Input dataset not found!")
        print(f"   Path: {INPUT_DATASET}")
        print(f"\n💡 Update INPUT_DATASET in the script")
        return
    
    # Confirm
    print(f"\n📋 Configuration:")
    print(f"   Input:  {INPUT_DATASET}")
    print(f"   Output: {OUTPUT_DATASET}")
    print(f"   Size:   {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"   Mode:   {TARGET_MODE} (3 channels)")
    
    response = input("\n🚀 Start preprocessing? (yes/no): ").strip().lower()
    
    if response == 'yes':
        # Preprocess
        output_path = create_preprocessed_dataset(
            INPUT_DATASET, 
            OUTPUT_DATASET,
            TARGET_SIZE,
            TARGET_MODE,
            QUALITY
        )
        
        # Verify
        print("\n")
        verify_preprocessed_dataset(output_path, TARGET_SIZE, TARGET_MODE)
        
        print("\n" + "="*80)
        print("🎉 ALL DONE!")
        print("="*80)
        print(f"\n📝 TODO:")
        print(f"   1. Update scripts/config.py:")
        print(f"      DATASET_ROOT = r\"{OUTPUT_DATASET}\"")
        print(f"\n   2. Your dataset is now optimized for training!")
        print(f"      - No runtime conversion needed")
        print(f"      - Faster data loading")
        print(f"      - Ready for MobileNetV2")
        
    else:
        print("\n❌ Cancelled by user")


if __name__ == "__main__":
    main()