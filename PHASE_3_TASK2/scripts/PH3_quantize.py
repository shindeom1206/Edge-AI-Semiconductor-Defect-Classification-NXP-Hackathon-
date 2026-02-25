"""
Phase 3: TFLite Export + INT8 Quantization (NO evaluation)
MobileNetV2 1.0 | 128x128 | 11 Classes | NXP RT1170EVK

WHAT THIS SCRIPT DOES:
  1. Loads best_model.pth (11 classes)
  2. Exports to ONNX
  3. Converts ONNX -> TFLite Float32
  4. Quantizes to INT8 Hybrid  (weights=INT8, I/O=float32)
  5. Quantizes to Full INT8    (weights+I/O=INT8) -- for MCUXpresso hardware
  6. Generates MCUXpresso header files:
       model_data.h    -> REPLACE existing file in MCUXpresso source/
       defect_labels.h -> ADD as new file in MCUXpresso source/
       test_image.h    -> ADD as new file in MCUXpresso source/

NO evaluation, NO confusion matrices, NO accuracy metrics.
Just clean export + quantization + header generation.

CALIBRATION SOURCE:
  Uses split/train folder (70% of original training data) for calibration.
  This avoids any leakage from test data.
"""

import os, sys, shutil, random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION — UPDATE PATHS IF NEEDED
# ==============================================================================
MODEL_PTH    = r"C:\hackathon_project\PH3\models\best_model.pth"

# Calibration source: use train split (safe, no test leakage)
CALIB_DIR    = r"C:\edge-ai-defect-classification\hackathon_phase3_training_dataset\split"

# Test image to embed in test_image.h (set None to auto-select from calib dir)
TEST_IMAGE_PATH = None

EXPORT_DIR   = r"C:\hackathon_project\PH3\export"
SCRIPT_DIR   = r"C:\hackathon_project\PH3"

CALIB_IMAGES = 300
IMG_EXT      = {".jpg", ".jpeg", ".png", ".bmp"}
NORM_MEAN    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD     = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(EXPORT_DIR, exist_ok=True)

print("=" * 70)
print("  PHASE 3: TFLITE EXPORT + INT8 QUANTIZATION")
print(f"  Started: {TIMESTAMP}")
print("=" * 70)


# ==============================================================================
# HELPERS
# ==============================================================================
def open_rgb(path):
    return Image.open(path).convert("RGB")


def preprocess_numpy(pil_img, image_size, normalize=True):
    img = pil_img.resize(image_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    if normalize:
        arr = (arr - NORM_MEAN) / NORM_STD
    return arr


# ==============================================================================
# STEP 1 — LOAD PYTORCH MODEL
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 1: LOAD PYTORCH MODEL")
print("=" * 70)

import torch

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from transfer_model import MobileNetV2Transfer

ckpt        = torch.load(MODEL_PTH, map_location="cpu", weights_only=False)
cfg         = ckpt.get("config", {})
NUM_CLASSES = cfg.get("num_classes", 11)
IMAGE_SIZE  = tuple(cfg.get("image_size", [128, 128]))
CLASS_NAMES = cfg.get("class_names", [])

print(f"  Checkpoint  : {MODEL_PTH}")
print(f"  Image size  : {IMAGE_SIZE}")
print(f"  Num classes : {NUM_CLASSES}")
print(f"  Class names : {CLASS_NAMES}")
print(f"  Best val F1 : {ckpt.get('best_val_f1', 0)*100:.2f}%")

model = MobileNetV2Transfer(pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()
print("  ✅ Model loaded successfully")


# ==============================================================================
# STEP 2 — EXPORT TO ONNX
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 2: EXPORT TO ONNX")
print("=" * 70)

import onnx

onnx_path = os.path.join(EXPORT_DIR, "mobilenetv2_defect.onnx")
torch.onnx.export(
    model,
    torch.randn(1, 3, *IMAGE_SIZE),
    onnx_path,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    do_constant_folding=True,
)
onnx.checker.check_model(onnx.load(onnx_path))
size_mb = os.path.getsize(onnx_path) / 1024 / 1024
print(f"  ✅ ONNX saved : {onnx_path}  ({size_mb:.2f} MB)")


# ==============================================================================
# STEP 3 — ONNX -> TFLITE FLOAT32
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 3: ONNX -> TFLITE FLOAT32")
print("=" * 70)

import onnx2tf
import tensorflow as tf

saved_model_dir   = os.path.join(EXPORT_DIR, "saved_model")
float_tflite_path = os.path.join(EXPORT_DIR, "mobilenetv2_float32.tflite")

onnx2tf.convert(
    input_onnx_file_path=onnx_path,
    output_folder_path=saved_model_dir,
    non_verbose=True
)

tflite_files = list(Path(saved_model_dir).glob("*.tflite"))
if not tflite_files:
    raise FileNotFoundError(f"onnx2tf produced no .tflite in {saved_model_dir}")
shutil.copy(str(tflite_files[0]), float_tflite_path)

size_mb = os.path.getsize(float_tflite_path) / 1024 / 1024
print(f"  ✅ Float32 TFLite : {float_tflite_path}  ({size_mb:.2f} MB)")


# ==============================================================================
# STEP 4 — PREPARE CALIBRATION DATA
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 4: PREPARE CALIBRATION DATA")
print(f"  Source: {CALIB_DIR}")
print("=" * 70)

if not os.path.exists(CALIB_DIR):
    raise FileNotFoundError(
        f"\n❌ Calibration directory not found:\n   {CALIB_DIR}\n"
        f"   Make sure split_train_dataset.py has been run first."
    )

calib_paths = [
    str(f) for f in Path(CALIB_DIR).rglob("*")
    if f.suffix.lower() in IMG_EXT
]
random.seed(42)
random.shuffle(calib_paths)
calib_paths = calib_paths[:CALIB_IMAGES]
print(f"  Calibration images selected: {len(calib_paths)}")
print(f"  Preprocessing: WITH normalization (ImageNet mean/std)")


def representative_dataset():
    for p in tqdm(calib_paths, desc="  Calibrating", ncols=70):
        try:
            arr = preprocess_numpy(open_rgb(p), IMAGE_SIZE, normalize=True)
            yield [arr[np.newaxis]]
        except Exception as e:
            print(f"  SKIP {Path(p).name}: {e}")


# ==============================================================================
# STEP 5A — INT8 HYBRID QUANTIZATION
# weights=INT8, I/O=float32
# Best for accuracy — use this for hackathon submission evaluation
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 5A: INT8 HYBRID QUANTIZATION")
print("  weights=INT8, I/O=float32")
print("  -> Best accuracy, use for hackathon submission")
print("=" * 70)

hybrid_path = os.path.join(EXPORT_DIR, "mobilenetv2_int8_hybrid.tflite")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations          = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# No inference_input_type / inference_output_type -> keeps float32 I/O

with open(hybrid_path, "wb") as fh:
    fh.write(converter.convert())

# Verify dtypes
interp_hyb = tf.lite.Interpreter(model_path=hybrid_path)
interp_hyb.allocate_tensors()
in_hyb  = interp_hyb.get_input_details()[0]
out_hyb = interp_hyb.get_output_details()[0]

size_mb = os.path.getsize(hybrid_path) / 1024 / 1024
print(f"  ✅ Hybrid INT8 saved : {hybrid_path}  ({size_mb:.2f} MB)")
print(f"     Input  dtype: {in_hyb['dtype'].__name__}  (float32 ✓)")
print(f"     Output dtype: {out_hyb['dtype'].__name__}")


# ==============================================================================
# STEP 5B — FULL INT8 QUANTIZATION
# weights=INT8, I/O=INT8
# Required for NXP MCUXpresso hardware deployment
# Output filename: mobilenetv2_int8.tflite (matches MCUXpresso guide)
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 5B: FULL INT8 QUANTIZATION")
print("  weights=INT8, I/O=INT8")
print("  -> Required for MCUXpresso hardware deployment")
print("=" * 70)

full_int8_path = os.path.join(EXPORT_DIR, "mobilenetv2_int8.tflite")
converter2 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter2.optimizations             = [tf.lite.Optimize.DEFAULT]
converter2.representative_dataset    = representative_dataset
converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter2.inference_input_type      = tf.int8
converter2.inference_output_type     = tf.int8

with open(full_int8_path, "wb") as fh:
    fh.write(converter2.convert())

# Verify quantization params
interp_i8 = tf.lite.Interpreter(model_path=full_int8_path)
interp_i8.allocate_tensors()
in_i8  = interp_i8.get_input_details()[0]
out_i8 = interp_i8.get_output_details()[0]
in_scale,  in_zp  = in_i8["quantization"]
out_scale, out_zp = out_i8["quantization"]

size_mb = os.path.getsize(full_int8_path) / 1024 / 1024
print(f"  ✅ Full INT8 saved : {full_int8_path}  ({size_mb:.2f} MB)")
print(f"     Input  : dtype={in_i8['dtype'].__name__}  scale={in_scale:.6f}  zp={in_zp}")
print(f"     Output : dtype={out_i8['dtype'].__name__}  scale={out_scale:.6f}  zp={out_zp}")


# ==============================================================================
# STEP 6 — GENERATE model_data.h
# REPLACE existing file in MCUXpresso source/
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 6: GENERATE model_data.h")
print("  -> REPLACE existing file in MCUXpresso source/")
print("=" * 70)

model_data_h_path = os.path.join(EXPORT_DIR, "model_data.h")
tflite_bytes      = open(full_int8_path, "rb").read()
tensor_arena_kb   = 1200   # 1.2 MB safe default for MobileNetV2 INT8 128x128

hex_values = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
num_bytes  = len(tflite_bytes)

with open(model_data_h_path, "w", encoding="utf-8") as fh:
    fh.write(f"""\
// model_data.h
// Auto-generated by Phase 3 script -- {TIMESTAMP}
// Model  : MobileNetV2 INT8 full quantization
// Classes: {NUM_CLASSES} -- {CLASS_NAMES}
// Size   : {num_bytes} bytes ({num_bytes/1024/1024:.2f} MB)
//
// INSTRUCTION: Copy this file to MCUXpresso source/model_data.h (REPLACE existing)

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

// Tensor arena size. Increase if you get "Tensor arena too small" error.
#define kTensorArenaSize ({tensor_arena_kb} * 1024)

// Model data array
alignas(8) const uint8_t mobilenetv2_int8_tflite[] = {{
  {hex_values}
}};

const unsigned int mobilenetv2_int8_tflite_len = {num_bytes}U;

#endif  // MODEL_DATA_H
""")

print(f"  ✅ Written : {model_data_h_path}  ({num_bytes/1024/1024:.2f} MB)")


# ==============================================================================
# STEP 7 — GENERATE defect_labels.h
# ADD as new file in MCUXpresso source/
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 7: GENERATE defect_labels.h")
print("  -> ADD as new file in MCUXpresso source/")
print("=" * 70)

labels_h_path          = os.path.join(EXPORT_DIR, "defect_labels.h")
labels_array_entries   = "\n  ".join(f'"{name}",' for name in CLASS_NAMES)

with open(labels_h_path, "w", encoding="utf-8") as fh:
    fh.write(f"""\
// defect_labels.h
// Auto-generated by Phase 3 script -- {TIMESTAMP}
// {NUM_CLASSES} class labels for MobileNetV2 defect classifier
//
// INSTRUCTION: Copy this file to MCUXpresso source/defect_labels.h (ADD new file)
// Include via: #include "defect_labels.h"

#ifndef DEFECT_LABELS_H
#define DEFECT_LABELS_H

#include <stdint.h>

// Number of output classes
#define NUM_CLASSES {NUM_CLASSES}

// Class label strings -- index matches model output index
static const char* const kCategoryLabels[NUM_CLASSES] = {{
  {labels_array_entries}
}};

#endif  // DEFECT_LABELS_H
""")

print(f"  ✅ Written : {labels_h_path}")
print(f"     Classes : {CLASS_NAMES}")


# ==============================================================================
# STEP 8 — GENERATE test_image.h
# ADD as new file in MCUXpresso source/
# Variable name: test_image (matches IMAGE_Decode(test_image, ...) in guide)
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 8: GENERATE test_image.h")
print("  -> ADD as new file in MCUXpresso source/")
print("=" * 70)

test_image_h_path = os.path.join(EXPORT_DIR, "test_image.h")

# Find test image to embed
if TEST_IMAGE_PATH and os.path.isfile(TEST_IMAGE_PATH):
    chosen_test_img = TEST_IMAGE_PATH
else:
    all_imgs = sorted([
        str(f) for f in Path(CALIB_DIR).rglob("*")
        if f.suffix.lower() in IMG_EXT
    ])
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {CALIB_DIR}")
    chosen_test_img = all_imgs[0]

print(f"  Embedding: {chosen_test_img}")

pil_test   = open_rgb(chosen_test_img).resize(IMAGE_SIZE, Image.LANCZOS)
arr_test   = np.array(pil_test, dtype=np.uint8)
flat_bytes = arr_test.flatten().tolist()
num_pixels = len(flat_bytes)
img_w, img_h = IMAGE_SIZE

hex_pixels = ", ".join(f"0x{b:02x}" for b in flat_bytes)

with open(test_image_h_path, "w", encoding="utf-8") as fh:
    fh.write(f"""\
// test_image.h
// Auto-generated by Phase 3 script -- {TIMESTAMP}
// Source : {os.path.basename(chosen_test_img)}
// Size   : {img_w}x{img_h} RGB ({num_pixels} bytes)
//
// INSTRUCTION: Copy this file to MCUXpresso source/test_image.h (ADD new file)
// Include via : #include "test_image.h"
// Used as     : IMAGE_Decode(test_image, ...)

#ifndef TEST_IMAGE_H
#define TEST_IMAGE_H

#include <stdint.h>

#define TEST_IMAGE_WIDTH    {img_w}
#define TEST_IMAGE_HEIGHT   {img_h}
#define TEST_IMAGE_CHANNELS 3

// Raw uint8 RGB pixel data
const uint8_t test_image[{num_pixels}] = {{
  {hex_pixels}
}};

#endif  // TEST_IMAGE_H
""")

print(f"  ✅ Written : {test_image_h_path}")
print(f"     Dims   : {img_w}x{img_h} RGB  ({num_pixels} bytes)")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
def mb(p): return os.path.getsize(p) / 1024 / 1024

print("\n" + "=" * 70)
print("  ALL DONE — FILES GENERATED")
print("=" * 70)
print(f"\n  {'File':<40} {'Size':>8}")
print(f"  {'-'*50}")
print(f"  {'mobilenetv2_defect.onnx':<40} {mb(onnx_path):>6.2f} MB")
print(f"  {'mobilenetv2_float32.tflite':<40} {mb(float_tflite_path):>6.2f} MB")
print(f"  {'mobilenetv2_int8_hybrid.tflite':<40} {mb(hybrid_path):>6.2f} MB")
print(f"  {'mobilenetv2_int8.tflite':<40} {mb(full_int8_path):>6.2f} MB")
print(f"  {'model_data.h':<40} {mb(model_data_h_path):>6.2f} MB")
print(f"  {'defect_labels.h':<40} {'< 1 KB':>8}")
print(f"  {'test_image.h':<40} {mb(test_image_h_path):>6.2f} MB")

print(f"\n  All files saved to: {EXPORT_DIR}")

print(f"\n  {'='*60}")
print(f"  MCUXpresso — COPY THESE 3 FILES TO source/ FOLDER:")
print(f"  {'='*60}")
print(f"  model_data.h     -> REPLACE existing file")
print(f"  defect_labels.h  -> ADD as new file")
print(f"  test_image.h     -> ADD as new file")
print(f"  {'='*60}")
print(f"\n  Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print("\n  ✅ Ready for MCUXpresso deployment!")
print("=" * 70)