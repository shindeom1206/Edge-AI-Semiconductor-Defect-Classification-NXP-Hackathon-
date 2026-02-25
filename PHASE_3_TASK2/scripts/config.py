"""
Configuration - Phase 3 FIXED
MobileNetV2 1.0 | 128x128 | 11 Classes | NXP RT1170EVK

WHY ACCURACY WAS 39% — ROOT CAUSES FIXED HERE:
  ❌ PHASE_2_EPOCHS = 40  → only ran ~5 epochs total in 1.8 min, model never converged
  ❌ LEARNING_RATE_PHASE2 = 1e-5  → too conservative, barely moved weights
  ❌ EARLY_STOP_PATIENCE = 10  → stopped training before model found good weights
  ❌ LR_PATIENCE = 3  → reduced LR too quickly, stunted learning
  ❌ LR_MIN = 1e-7  → hit floor too early

FIXES:
  ✅ PHASE_1_EPOCHS: 15 → 20   (more warmup for classifier head)
  ✅ PHASE_2_EPOCHS: 40 → 80   (main fix — needs much more fine-tuning time)
  ✅ LEARNING_RATE_PHASE2: 1e-5 → 5e-5  (3x higher, actually moves weights)
  ✅ EARLY_STOP_PATIENCE: 10 → 20  (don't quit too early)
  ✅ LR_PATIENCE: 3 → 5  (let val loss stabilise before reducing)
  ✅ LR_MIN: 1e-7 → 1e-8  (allow deeper fine-tuning at end)
  ✅ WEIGHT_DECAY_PHASE2: 2e-4 → 1e-4  (less regularisation during fine-tune)

Expected result after fix: 65-80% accuracy (up from 39%)

DATASET:
  Run split_train_dataset.py first to create the split/ folder.
  The 11 class folders from Hackathon_phase3_training_dataset are split:
    70% -> split/train  |  15% -> split/val  |  15% -> split/test

CLASS INDEX MAPPING (strict alphabetical — matches ImageFolder):
  0=BRIDGE  1=CLEAN_CRACK  2=CLEAN_LAYER  3=CLEAN_VIA  4=CMP
  5=CRACK   6=LER          7=OPEN         8=OTHERS     9=PARTICLE  10=VIA
"""

import os

# ============================================================================
# PATHS — DO NOT CHANGE
# ============================================================================
PROJECT_ROOT = r"C:\hackathon_project\PH3"
SPLIT_ROOT   = r"C:\edge-ai-defect-classification\Hackathon_phase3_training_dataset\split"

TRAIN_DIR      = os.path.join(SPLIT_ROOT, "train")
VAL_DIR        = os.path.join(SPLIT_ROOT, "val")
TEST_DIR       = os.path.join(SPLIT_ROOT, "test")

MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR       = os.path.join(PROJECT_ROOT, "logs")

# ============================================================================
# DATA
# ============================================================================
IMAGE_SIZE  = (128, 128)
NUM_CLASSES = 11

CLASS_NAMES = [
    'BRIDGE', 'CLEAN_CRACK', 'CLEAN_LAYER', 'CLEAN_VIA',
    'CMP', 'CRACK', 'LER', 'OPEN', 'OTHERS', 'PARTICLE', 'VIA'
]

# ============================================================================
# MODEL
# ============================================================================
BACKBONE             = "mobilenet_v2"
PRETRAINED           = True
WIDTH_MULT           = 1.0
MOBILENET_WIDTH_MULT = WIDTH_MULT

DROPOUT_RATE_1 = 0.4
DROPOUT_RATE_2 = 0.2

# ============================================================================
# PHASE 1 — Classifier only (backbone frozen)
# FIX: 20 epochs (was 15) — more warmup time for the classifier head
# ============================================================================
PHASE_1_EPOCHS       = 45
LEARNING_RATE_PHASE1 = 1e-3
BATCH_SIZE           = 32
WEIGHT_DECAY_PHASE1  = 1e-4

# ============================================================================
# PHASE 2 — Full fine-tuning (backbone unfrozen)
# FIX: 80 epochs (was 40) — THIS IS THE MAIN FIX. Only 5 epochs ran before.
# FIX: LR = 5e-5 (was 1e-5) — 5x higher so weights actually update meaningfully
# FIX: WEIGHT_DECAY = 1e-4 (was 2e-4) — less regularisation during fine-tune
# ============================================================================
PHASE_2_EPOCHS       = 105
LEARNING_RATE_PHASE2 = 5e-5
BATCH_SIZE_PHASE2    = 32
WEIGHT_DECAY_PHASE2  = 1e-4

# ============================================================================
# REGULARISATION
# FIX: EARLY_STOP_PATIENCE = 20 (was 10) — stopped killing training too early
# ============================================================================
MAX_GRAD_NORM              = 1.0
EARLY_STOP_PATIENCE        = 20
OVERFIT_WARNING_THRESHOLD  = 0.10
OVERFIT_CRITICAL_THRESHOLD = 0.15

# ============================================================================
# LR SCHEDULER
# FIX: LR_PATIENCE = 5 (was 3) — don't reduce LR so aggressively
# FIX: LR_MIN = 1e-8 (was 1e-7) — allow deeper convergence late in training
# ============================================================================
LR_SCHEDULER = "ReduceLROnPlateau"
LR_FACTOR    = 0.5
LR_PATIENCE  = 5
LR_MIN       = 1e-8

# ============================================================================
# AUGMENTATION — unchanged (these were correct)
# ============================================================================
USE_AUGMENTATION        = True
RANDOM_HORIZONTAL_FLIP  = 0.5
RANDOM_VERTICAL_FLIP    = 0.5
RANDOM_ROTATION_DEGREES = 180
COLOR_JITTER_BRIGHTNESS = 0.3
COLOR_JITTER_CONTRAST   = 0.3
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE        = 0.05
RANDOM_AFFINE_DEGREES   = 15
RANDOM_AFFINE_TRANSLATE = (0.1, 0.1)
RANDOM_AFFINE_SCALE     = (0.9, 1.1)
USE_RANDOM_ERASING      = True
RANDOM_ERASING_PROB     = 0.3
RANDOM_ERASING_SCALE    = (0.02, 0.33)

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ============================================================================
# HARDWARE
# ============================================================================
USE_GPU             = True
NUM_WORKERS         = 0
PIN_MEMORY          = True
USE_MIXED_PRECISION = False

# ============================================================================
# LOGGING / SAVING
# ============================================================================
SAVE_BEST_ONLY = True
SAVE_FREQUENCY = 5
LOG_FREQUENCY  = 10
VERBOSE        = True

# ============================================================================
# EVALUATION
# ============================================================================
TRACK_ACCURACY   = True
TRACK_PRECISION  = True
TRACK_RECALL     = True
TRACK_F1_SCORE   = True
METRIC_AVERAGE   = "macro"

# ============================================================================
# MISC
# ============================================================================
RANDOM_SEED            = 42
PLOT_TRAINING_CURVES   = True
PLOT_CONFUSION_MATRIX  = True
PLOT_DPI               = 300
USE_CLASS_WEIGHTS      = True
USE_LABEL_SMOOTHING    = False
LABEL_SMOOTHING_FACTOR = 0.1
USE_MIXUP              = False
MIXUP_ALPHA            = 0.2
DEBUG_MODE             = False
QUICK_RUN              = False

if QUICK_RUN:
    PHASE_1_EPOCHS      = 2
    PHASE_2_EPOCHS      = 2
    EARLY_STOP_PATIENCE = 2

# ============================================================================
# DISPLAY
# ============================================================================
def print_config():
    print("\n" + "="*65)
    print("  PHASE 3 CONFIG — FIXED (11 Classes)")
    print("="*65)
    print(f"  Backbone:   MobileNetV2 {WIDTH_MULT} (ImageNet pretrained)")
    print(f"  Image:      {IMAGE_SIZE}")
    print(f"  Classes:    {NUM_CLASSES}  →  {CLASS_NAMES}")
    print(f"  Phase 1:    {PHASE_1_EPOCHS} epochs  LR={LEARNING_RATE_PHASE1}  [classifier only]")
    print(f"  Phase 2:    {PHASE_2_EPOCHS} epochs  LR={LEARNING_RATE_PHASE2}  [full fine-tune]  ← main fix")
    print(f"  Max total:  {PHASE_1_EPOCHS + PHASE_2_EPOCHS} epochs")
    print(f"  Early stop: patience={EARLY_STOP_PATIENCE}  ← was 10, now 20")
    print(f"  LR sched:   {LR_SCHEDULER}  factor={LR_FACTOR}  patience={LR_PATIENCE}  min={LR_MIN}")
    print(f"  Class wts:  {USE_CLASS_WEIGHTS}")
    print(f"  Est. time:  ~25-40 min (GPU) / ~3-5 hrs (CPU)")
    print(f"  Train dir:  {TRAIN_DIR}")
    print(f"  Val dir:    {VAL_DIR}")
    print(f"  Test dir:   {TEST_DIR}")
    print("="*65 + "\n")

if __name__ == "__main__":
    print_config()