"""
Configuration File - Optimized for Anti-Overfitting
"""

import os

DATASET_ROOT = "C:/edge-ai-defect-classification/dataset_128"
PROJECT_ROOT = "C:/hackathon_project"

DATA_DIR = DATASET_ROOT
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 8  # ✅ Correct for your 8 classes
CLASS_NAMES = ['LER', 'bridge', 'clean', 'crack', 'open', 'other', 'particle', 'scratch']  # ✅ Perfect!

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
BACKBONE = "mobilenet_v2"
PRETRAINED = True  # Use ImageNet pre-trained weights

# Dropout rates (INCREASED to prevent overfitting)
DROPOUT_RATE_1 = 0.6  # Increased from typical 0.2-0.3
DROPOUT_RATE_2 = 0.4  # Additional dropout layer

# ============================================================================
# TRAINING HYPERPARAMETERS - PHASE 1 (Classifier Only)
# ============================================================================
PHASE_1_EPOCHS = 20
LEARNING_RATE_PHASE1 = 1e-3  # Standard for classifier training
BATCH_SIZE = 32  # Adjust based on GPU memory

# ============================================================================
# TRAINING HYPERPARAMETERS - PHASE 2 (Fine-tuning)
# ============================================================================
PHASE_2_EPOCHS = 30
LEARNING_RATE_PHASE2 = 1e-5  # REDUCED from 1e-4 to prevent overfitting
BATCH_SIZE_PHASE2 = 32  # Can be same or different

# ============================================================================
# REGULARIZATION (Anti-Overfitting Measures)
# ============================================================================
# Weight decay (L2 regularization) - Applied in optimizer
WEIGHT_DECAY_PHASE1 = 1e-4
WEIGHT_DECAY_PHASE2 = 2e-4  # Stronger in phase 2

# Gradient clipping
MAX_GRAD_NORM = 1.0

# Early stopping
EARLY_STOP_PATIENCE = 7  # Stop after 7 epochs without improvement

# Overfitting monitoring thresholds
OVERFIT_WARNING_THRESHOLD = 0.10   # 10% train-val gap = warning
OVERFIT_CRITICAL_THRESHOLD = 0.15  # 15% train-val gap = critical

# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================
LR_SCHEDULER = "ReduceLROnPlateau"
LR_FACTOR = 0.5      # Reduce LR by half when plateau
LR_PATIENCE = 3      # Wait 3 epochs before reducing LR
LR_MIN = 1e-7        # Minimum learning rate

# ============================================================================
# DATA AUGMENTATION (Anti-Overfitting)
# ============================================================================
# Training augmentations
USE_AUGMENTATION = True

# Basic augmentations
RANDOM_HORIZONTAL_FLIP = 0.5
RANDOM_VERTICAL_FLIP = 0.3
RANDOM_ROTATION_DEGREES = 20

# Color augmentations
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.1

# Geometric augmentations
RANDOM_AFFINE_DEGREES = 15
RANDOM_AFFINE_TRANSLATE = (0.1, 0.1)
RANDOM_AFFINE_SCALE = (0.9, 1.1)

# Advanced augmentations (optional)
USE_RANDOM_ERASING = True
RANDOM_ERASING_PROB = 0.3
RANDOM_ERASING_SCALE = (0.02, 0.33)

# Normalization (ImageNet statistics for transfer learning)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ============================================================================
# HARDWARE SETTINGS
# ============================================================================
USE_GPU = True
NUM_WORKERS = 4  # Data loading workers (set to 0 if issues on Windows)
PIN_MEMORY = True

# Mixed precision training (faster, less memory)
USE_MIXED_PRECISION = False  # Set True if GPU supports it (RTX cards)

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================
SAVE_BEST_ONLY = True
SAVE_FREQUENCY = 5  # Save checkpoint every N epochs
LOG_FREQUENCY = 10  # Log metrics every N batches

VERBOSE = True

# ============================================================================
# EVALUATION METRICS
# ============================================================================
# Which metrics to track
TRACK_ACCURACY = True
TRACK_PRECISION = True
TRACK_RECALL = True
TRACK_F1_SCORE = True

# Averaging method for multi-class metrics
METRIC_AVERAGE = 'macro'  # 'macro', 'weighted', or 'micro'

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# VISUALIZATION
# ============================================================================
PLOT_TRAINING_CURVES = True
PLOT_CONFUSION_MATRIX = True
PLOT_DPI = 300

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================
# Label smoothing (helps with overfitting)
USE_LABEL_SMOOTHING = False
LABEL_SMOOTHING_FACTOR = 0.1

# Mixup augmentation (advanced, optional)
USE_MIXUP = False
MIXUP_ALPHA = 0.2

# Class balancing
USE_CLASS_WEIGHTS = True  # Automatically computed from dataset

# ============================================================================
# DEBUGGING
# ============================================================================
DEBUG_MODE = False
QUICK_RUN = False  # Use small subset for testing

if QUICK_RUN:
    PHASE_1_EPOCHS = 2
    PHASE_2_EPOCHS = 2
    EARLY_STOP_PATIENCE = 1

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================
def print_config():
    """Print configuration summary"""
    print("\n" + "="*70)
    print("📋 CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\n🎯 MODEL:")
    print(f"   Backbone: {BACKBONE}")
    print(f"   Classes: {NUM_CLASSES}")
    print(f"   Image size: {IMAGE_SIZE}")
    print(f"   Dropout: {DROPOUT_RATE_1} / {DROPOUT_RATE_2}")
    
    print(f"\n🏋️  TRAINING:")
    print(f"   Phase 1: {PHASE_1_EPOCHS} epochs @ LR={LEARNING_RATE_PHASE1}")
    print(f"   Phase 2: {PHASE_2_EPOCHS} epochs @ LR={LEARNING_RATE_PHASE2}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    print(f"\n🛡️  REGULARIZATION:")
    print(f"   Weight Decay P1: {WEIGHT_DECAY_PHASE1}")
    print(f"   Weight Decay P2: {WEIGHT_DECAY_PHASE2}")
    print(f"   Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"   Gradient Clipping: {MAX_GRAD_NORM}")
    
    print(f"\n📊 DATA AUGMENTATION:")
    print(f"   Enabled: {USE_AUGMENTATION}")
    print(f"   Horizontal Flip: {RANDOM_HORIZONTAL_FLIP}")
    print(f"   Rotation: ±{RANDOM_ROTATION_DEGREES}°")
    print(f"   Color Jitter: {COLOR_JITTER_BRIGHTNESS}")
    
    print(f"\n💾 PATHS:")
    print(f"   Dataset Root: {DATASET_ROOT}")
    print(f"   Train: {TRAIN_DIR}")
    print(f"   Val: {VAL_DIR}")
    print(f"   Test: {TEST_DIR}")
    print(f"   Models: {MODEL_SAVE_DIR}")
    print(f"   Results: {RESULTS_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    print_config()
