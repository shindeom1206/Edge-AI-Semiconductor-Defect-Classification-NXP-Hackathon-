"""
Hackathon Phase-3 Prediction Script
FP32 TFLite | MobileNetV2 1.0 | 128x128 | 11 Classes
NXP Hackathon Phase-3 Task-2
"""

import os
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION  <-- edit paths if needed
# ==============================================================================
MODEL_TFLITE = r"C:\hackathon_project\PH3\export\mobilenetv2_float32.tflite"
PREDICT_DIR  = r"C:\edge-ai-defect-classification\Hackathon_phase3_prediction_dataset"
RESULTS_DIR  = r"C:\hackathon_project\PH3\results"

# 11 classes -- strict alphabetical order (matches ImageFolder / config.py)
CLASS_NAMES = [
    'BRIDGE', 'CLEAN_CRACK', 'CLEAN_LAYER', 'CLEAN_VIA',
    'CMP', 'CRACK', 'LER', 'OPEN', 'OTHERS', 'PARTICLE', 'VIA'
]

IMAGE_SIZE = (128, 128)   # auto-detected from model -- this is just fallback
IMG_EXT    = {".jpg", ".jpeg", ".png", ".bmp"}

NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("  HACKATHON PHASE-3 PREDICTION")
print("  NXP Hackathon | MobileNetV2 INT8 | 11 Classes | No Aug | No TTA")
print(f"  Started: {TIMESTAMP}")
print("=" * 80)

# ==============================================================================
# LOAD TFLITE MODEL
# ==============================================================================
import tensorflow as tf

if not os.path.exists(MODEL_TFLITE):
    print(f"\n  ERROR: Model not found: {MODEL_TFLITE}")
    raise SystemExit(1)

interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

# Auto-detect size from model
H = int(in_det['shape'][1])
W = int(in_det['shape'][2])
IMAGE_SIZE = (W, H)

print(f"\n  Model       : {MODEL_TFLITE}")
print(f"  Input shape : {in_det['shape']}  dtype: {in_det['dtype'].__name__}")
print(f"  Output shape: {out_det['shape']}  dtype: {out_det['dtype'].__name__}")
print(f"  Image size  : {IMAGE_SIZE}")
print(f"  Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

# ==============================================================================
# PREPROCESS -- grayscale safe, no augmentation
# ==============================================================================
def preprocess(path):
    # .convert("RGB") replicates single b/w channel to 3 -- not augmentation
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    return arr[np.newaxis].astype(np.float32)   # [1, H, W, 3]

# ==============================================================================
# COLLECT ALL IMAGES
# ==============================================================================
if not os.path.exists(PREDICT_DIR):
    print(f"\n  ERROR: Prediction directory not found: {PREDICT_DIR}")
    raise SystemExit(1)

all_images = sorted([
    f for f in Path(PREDICT_DIR).rglob("*")
    if f.suffix.lower() in IMG_EXT
])

print(f"\n  Found {len(all_images)} images in: {PREDICT_DIR}")

if len(all_images) == 0:
    print("  ERROR: No images found. Check PREDICT_DIR path.")
    raise SystemExit(1)

# ==============================================================================
# RUN INFERENCE
# ==============================================================================
print()
results = []   # list of (image_name, predicted_class)
skipped = 0

for img_path in tqdm(all_images, desc="  Predicting", ncols=70):
    try:
        inp  = preprocess(str(img_path))
        interpreter.set_tensor(in_det["index"], inp)
        interpreter.invoke()
        out        = interpreter.get_tensor(out_det["index"])
        pred_idx   = int(np.argmax(out))
        pred_class = CLASS_NAMES[pred_idx]
        results.append((img_path.name, pred_class))
    except Exception as e:
        skipped += 1
        print(f"  SKIP {img_path.name}: {e}")
        results.append((img_path.name, "ERROR"))

print(f"\n  Predicted : {len(results) - skipped}")
print(f"  Skipped   : {skipped}")

# ==============================================================================
# PREDICTION DISTRIBUTION
# ==============================================================================
pred_counts = Counter(cls for _, cls in results)
print("\n  PREDICTION DISTRIBUTION:")
print("  " + "-" * 45)
for cls in CLASS_NAMES:
    count = pred_counts.get(cls, 0)
    bar   = "█" * count
    print(f"  {cls:<14}: {count:>4}  {bar}")

# ==============================================================================
# SAVE OUTPUT  -- exact hackathon format: "Image_name, Predicted_Class"
# ==============================================================================
out_path = os.path.join(RESULTS_DIR, f"predictions_{TIMESTAMP}.txt")

with open(out_path, "w", encoding="utf-8") as fh:
    for img_name, pred_class in results:
        fh.write(f"{img_name}, {pred_class}\n")

print(f"\n  Saved: {out_path}")
print(f"  Lines: {len(results)}")

# ==============================================================================
# PREVIEW
# ==============================================================================
print("\n  PREVIEW (first 10 lines):")
print("  " + "-" * 55)
print(f"  {'Image Name':<35} {'Predicted Class'}")
print("  " + "-" * 55)
for img_name, pred_class in results[:10]:
    print(f"  {img_name:<35} {pred_class}")
if len(results) > 10:
    print(f"  ... and {len(results) - 10} more lines")

print("\n" + "=" * 80)
print("  DONE  --  Upload this file to the hackathon portal:")
print(f"  {out_path}")
print("=" * 80)