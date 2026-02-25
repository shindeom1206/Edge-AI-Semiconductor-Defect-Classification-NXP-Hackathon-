"""
Data Loader - Phase 3 FINAL
MobileNetV2 1.0 | 128x128 | 11 Classes | NXP RT1170EVK

Uses standard ImageFolder on split/ folder which already has
11 class subfolders in train/, val/, test/.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

import config


# ============================================================================
class GrayscaleToRGB:
    """Convert any non-RGB image to RGB. Must be FIRST in pipeline."""
    def __call__(self, img):
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


# ============================================================================
def get_transforms(train=True):
    """
    Train:    GrayscaleToRGB → Resize → RandomCrop → Flips → Rotation
              → ColorJitter → RandomAffine → ToTensor → RandomErasing → Normalize
    Val/Test: GrayscaleToRGB → Resize → ToTensor → Normalize
    """
    if train and config.USE_AUGMENTATION:
        return transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize(config.IMAGE_SIZE),
            transforms.RandomCrop(config.IMAGE_SIZE, padding=8),
            transforms.RandomHorizontalFlip(p=config.RANDOM_HORIZONTAL_FLIP),
            transforms.RandomVerticalFlip(p=config.RANDOM_VERTICAL_FLIP),
            transforms.RandomRotation(degrees=config.RANDOM_ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=config.COLOR_JITTER_BRIGHTNESS,
                contrast=config.COLOR_JITTER_CONTRAST,
                saturation=config.COLOR_JITTER_SATURATION,
                hue=config.COLOR_JITTER_HUE
            ),
            transforms.RandomAffine(
                degrees=config.RANDOM_AFFINE_DEGREES,
                translate=config.RANDOM_AFFINE_TRANSLATE,
                scale=config.RANDOM_AFFINE_SCALE
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=config.RANDOM_ERASING_PROB,
                scale=config.RANDOM_ERASING_SCALE
            ) if config.USE_RANDOM_ERASING else transforms.Lambda(lambda x: x),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
    else:
        return transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])


# ============================================================================
def compute_class_weights(train_dataset):
    """Inverse-frequency weights for CrossEntropyLoss."""
    num_classes  = len(train_dataset.classes)
    class_counts = np.zeros(num_classes)

    for _, label in train_dataset:
        class_counts[label] += 1

    total   = len(train_dataset)
    weights = total / (num_classes * class_counts)
    weights = weights / weights.sum() * num_classes

    print("\n  Class distribution (train):")
    for i, (name, count, w) in enumerate(
            zip(train_dataset.classes, class_counts, weights)):
        print(f"    {i:>2}: {name:<14} | {int(count):>5} images | weight: {w:.3f}")

    return torch.FloatTensor(weights)


# ============================================================================
def get_data_loaders():
    """
    Create train / val / test DataLoaders from split/ folder.
    All 3 splits have the same 11 class folders.
    """
    print("\n" + "="*60)
    print("  CREATING DATA LOADERS")
    print("="*60)
    print(f"  Image size:   {config.IMAGE_SIZE}")
    print(f"  Classes:      {config.NUM_CLASSES}")
    print(f"  Class names:  {config.CLASS_NAMES}")
    print(f"  Batch size:   {config.BATCH_SIZE}")
    print(f"  Augmentation: {config.USE_AUGMENTATION}")

    train_transform = get_transforms(train=True)
    val_transform   = get_transforms(train=False)

    # ── TRAIN ────────────────────────────────────────────────────────────────
    if not os.path.exists(config.TRAIN_DIR):
        raise FileNotFoundError(
            f"\n❌ Train dir not found: {config.TRAIN_DIR}\n"
            f"   Run split_train_dataset.py first!"
        )
    train_dataset = datasets.ImageFolder(
        root=config.TRAIN_DIR, transform=train_transform
    )

    # ── VAL ──────────────────────────────────────────────────────────────────
    if not os.path.exists(config.VAL_DIR):
        raise FileNotFoundError(
            f"\n❌ Val dir not found: {config.VAL_DIR}\n"
            f"   Run split_train_dataset.py first!"
        )
    val_dataset = datasets.ImageFolder(
        root=config.VAL_DIR, transform=val_transform
    )

    # ── TEST ─────────────────────────────────────────────────────────────────
    if not os.path.exists(config.TEST_DIR):
        raise FileNotFoundError(
            f"\n❌ Test dir not found: {config.TEST_DIR}\n"
            f"   Run split_train_dataset.py first!"
        )
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR, transform=val_transform
    )

    print(f"\n  Datasets loaded:")
    print(f"    Train: {len(train_dataset):>5} images  →  {config.TRAIN_DIR}")
    print(f"    Val:   {len(val_dataset):>5} images  →  {config.VAL_DIR}")
    print(f"    Test:  {len(test_dataset):>5} images  →  {config.TEST_DIR}")

    # ── Verify class order ────────────────────────────────────────────────────
    dataset_classes = train_dataset.classes
    print(f"\n  Class order verification:")
    mismatch = False
    for i, cls in enumerate(dataset_classes):
        expected = config.CLASS_NAMES[i] if i < len(config.CLASS_NAMES) else "?"
        status   = "✓" if cls == expected else "✗ MISMATCH"
        if cls != expected:
            mismatch = True
        print(f"    {i:>2}: {cls:<14} config: {expected:<14} {status}")

    if mismatch:
        print(f"\n  ⚠️  Mismatch — updating config.CLASS_NAMES to match folders.")
        config.CLASS_NAMES = dataset_classes
        config.NUM_CLASSES = len(dataset_classes)
    else:
        print(f"\n  ✓ All {len(dataset_classes)} classes verified.")

    # ── Class weights ─────────────────────────────────────────────────────────
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_dataset)
    else:
        class_weights = torch.ones(config.NUM_CLASSES)
        print("\n  Using equal class weights.")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY if config.USE_GPU else False,
        drop_last=True
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

    print(f"\n  DataLoaders ready:")
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches:   {len(val_loader)}")
    print(f"    Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_weights


# ============================================================================
if __name__ == "__main__":
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    train_loader, val_loader, test_loader, weights = get_data_loaders()
    images, labels = next(iter(train_loader))
    print(f"\nSample batch: images={images.shape}  labels={labels.shape}")
    print(f"Unique labels: {sorted(labels.unique().tolist())}")
    print(f"Pixel range:   [{images.min():.3f}, {images.max():.3f}]")
    print("✅ Data loader test PASSED")