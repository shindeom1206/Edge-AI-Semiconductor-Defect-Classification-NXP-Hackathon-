""""
Data Loader for Transfer Learning
Compatible with new config format
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import os
from PIL import Image

import config


def get_transforms(train=True):
    """
    Get data transforms for training or validation/test
    
    Args:
        train (bool): If True, apply augmentations for training
    
    Returns:
        transforms.Compose: Composed transforms
    """
    
    if train and config.USE_AUGMENTATION:
        # TRAINING TRANSFORMS - WITH AUGMENTATION
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(config.IMAGE_SIZE),
            
            # Flip augmentations
            transforms.RandomHorizontalFlip(p=config.RANDOM_HORIZONTAL_FLIP),
            transforms.RandomVerticalFlip(p=config.RANDOM_VERTICAL_FLIP),
            
            # Rotation
            transforms.RandomRotation(degrees=config.RANDOM_ROTATION_DEGREES),
            
            # Color jitter
            transforms.ColorJitter(
                brightness=config.COLOR_JITTER_BRIGHTNESS,
                contrast=config.COLOR_JITTER_CONTRAST,
                saturation=config.COLOR_JITTER_SATURATION,
                hue=config.COLOR_JITTER_HUE
            ),
            
            # Affine transformations
            transforms.RandomAffine(
                degrees=config.RANDOM_AFFINE_DEGREES,
                translate=config.RANDOM_AFFINE_TRANSLATE,
                scale=config.RANDOM_AFFINE_SCALE
            ),
            
            # Convert to tensor
            transforms.ToTensor(),
            
            # Random erasing (optional, applied after ToTensor)
            transforms.RandomErasing(
                p=config.RANDOM_ERASING_PROB,
                scale=config.RANDOM_ERASING_SCALE
            ) if config.USE_RANDOM_ERASING else transforms.Lambda(lambda x: x),
            
            # Normalize using ImageNet statistics
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])
    else:
        # VALIDATION/TEST TRANSFORMS - NO AUGMENTATION
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])
    
    return transform


def compute_class_weights(train_dataset):
    """
    Compute class weights for imbalanced datasets
    
    Args:
        train_dataset: Training dataset
    
    Returns:
        torch.Tensor: Class weights
    """
    
    # Count samples per class
    class_counts = np.zeros(config.NUM_CLASSES)
    
    for _, label in train_dataset:
        class_counts[label] += 1
    
    # Compute weights (inverse frequency)
    total_samples = len(train_dataset)
    class_weights = total_samples / (config.NUM_CLASSES * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES
    
    print("\n📊 Class Distribution:")
    for i, (name, count, weight) in enumerate(zip(config.CLASS_NAMES, class_counts, class_weights)):
        print(f"   {name:12s}: {int(count):5d} samples (weight: {weight:.3f})")
    
    return torch.FloatTensor(class_weights)


class GrayscaleToRGB:
    """Convert grayscale images to RGB by duplicating channels"""
    
    def __call__(self, img):
        if img.mode == 'L':  # Grayscale
            return img.convert('RGB')
        return img


def get_data_loaders():
    """
    Create data loaders for train, validation, and test sets
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """
    
    print("\n🔄 Creating data loaders for transfer learning...")
    
    # Transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    test_transform = get_transforms(train=False)
    
    # Add grayscale to RGB conversion if needed
    # This ensures compatibility with MobileNetV2 which expects 3 channels
    train_transform.transforms.insert(0, GrayscaleToRGB())
    val_transform.transforms.insert(0, GrayscaleToRGB())
    test_transform.transforms.insert(0, GrayscaleToRGB())
    
    # Load datasets
    try:
        train_dataset = datasets.ImageFolder(
            root=config.TRAIN_DIR,
            transform=train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            root=config.VAL_DIR,
            transform=val_transform
        )
        
        test_dataset = datasets.ImageFolder(
            root=config.TEST_DIR,
            transform=test_transform
        )
        
        print(f"✅ Loaded datasets:")
        print(f"   Train: {len(train_dataset)} images")
        print(f"   Val:   {len(val_dataset)} images")
        print(f"   Test:  {len(test_dataset)} images")
        
        # Verify class names match
        dataset_classes = train_dataset.classes
        print(f"\n📂 Dataset classes (in order):")
        for i, cls in enumerate(dataset_classes):
            print(f"   {i}: {cls}")
        
        # Warning if mismatch
        if dataset_classes != config.CLASS_NAMES:
            print(f"\n⚠️  WARNING: Config class names don't match dataset!")
            print(f"   Config:  {config.CLASS_NAMES}")
            print(f"   Dataset: {dataset_classes}")
            print(f"   Using dataset order...")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find data directories!")
        print(f"   Train dir: {config.TRAIN_DIR}")
        print(f"   Val dir:   {config.VAL_DIR}")
        print(f"   Test dir:  {config.TEST_DIR}")
        raise e
    
    # Compute class weights for imbalanced data
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_dataset)
    else:
        class_weights = torch.ones(config.NUM_CLASSES)
        print("\n⚖️  Using equal class weights (no balancing)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY if config.USE_GPU else False,
        drop_last=True  # Drop incomplete batches for BatchNorm stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY if config.USE_GPU else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY if config.USE_GPU else False
    )
    
    print(f"\n✅ Data loaders created")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, class_weights


def visualize_batch(loader, num_images=8):
    """
    Visualize a batch of images with their labels
    
    Args:
        loader: DataLoader
        num_images: Number of images to show
    """
    
    import matplotlib.pyplot as plt
    
    # Get one batch
    images, labels = next(iter(loader))
    
    # Denormalize images
    mean = torch.tensor(config.NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(config.NORMALIZE_STD).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        label = labels[i].item()
        class_name = config.CLASS_NAMES[label]
        
        axes[i].imshow(img)
        axes[i].set_title(f"{class_name} (Class {label})")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "batch_preview.png"), dpi=150)
    print(f"\n📸 Saved batch preview to: {config.RESULTS_DIR}/batch_preview.png")
    plt.close()


def test_data_loader():
    """Test the data loader"""
    
    print("\n" + "="*70)
    print("🧪 TESTING DATA LOADER")
    print("="*70)
    
    try:
        train_loader, val_loader, test_loader, class_weights = get_data_loaders()
        
        print("\n✅ Data loader test successful!")
        
        # Test one batch
        images, labels = next(iter(train_loader))
        print(f"\n📦 Sample batch:")
        print(f"   Images shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Visualize
        if os.path.exists(config.RESULTS_DIR):
            visualize_batch(train_loader)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Data loader test failed!")
        print(f"   Error: {e}")
        return False


if __name__ == "__main__":
    # Create results directory if needed
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Test
    test_data_loader()