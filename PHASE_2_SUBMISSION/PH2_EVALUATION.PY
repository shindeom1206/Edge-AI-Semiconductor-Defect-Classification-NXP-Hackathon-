"""
hackathon_test_dataset_prediction.py
================================================================================
PHASE 2 - FINAL SUBMISSION
Model:      MobileNetV2 Transfer Learning (ONNX - Phase 1 submitted model)
Framework:  ONNX Runtime (same model as Phase 1 submission)
Target:     NXP i.MX RT (eIQ)

Rules compliance:
  - Using Phase 1 ONNX model exactly as submitted (no retraining, no re-export)
  - Preprocessing: Resize to 224x224 only (per organizer rules)
  - No TTA, no image enhancement (per organizer email Feb 16 2026)
  - CMP mapped to scratch (organizer confirmed: technically similar)
  - VIA mapped to other  (no matching training class)

Preprocessing pipeline matches training EXACTLY:
  Grayscale(1ch) -> convert('RGB') -> Resize BICUBIC 224x224 -> ToTensor -> Normalize
================================================================================
"""

import os
import sys
import json
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# CONFIGURATION  — all values hardcoded from model_metadata.json
# ============================================================================

class Config:
    # ---- Paths ----
    ONNX_MODEL_PATH = r"C:\hackathon_project\models\onnx\defect_classification_model.onnx"
    TEST_DATA_DIR   = r"C:\edge-ai-defect-classification\dataset_224_rgb\hackathon_test_dataset"
    OUTPUT_DIR      = r"C:\hackathon_project\PH2_FINAL_ONNX1"

    # ---- Model config (from model_metadata.json) ----
    # Input shape: [1, 3, 224, 224]
    IMG_SIZE        = (224, 224)
    BATCH_SIZE      = 16
    NUM_WORKERS     = 0   # Windows safe

    # Class names — EXACT order from training checkpoint
    # (verified from data.pkl inside best_model.pth)
    TRAINING_CLASSES = [
        'LER',       # index 0
        'bridge',    # index 1
        'clean',     # index 2
        'crack',     # index 3
        'open',      # index 4
        'other',     # index 5
        'particle',  # index 6
        'scratch',   # index 7
    ]

    # Hackathon test dataset folder names (9 folders — 2 not in training)
    TEST_CLASSES = [
        'Bridge',
        'CMP',
        'Clean',
        'Crack',
        'LER',
        'Open',
        'Other',
        'Particle',
        'VIA',
    ]

    # Class mapping overrides
    # CMP -> scratch  (organizer confirmed Feb 16 2026: technically similar to scratch)
    # VIA -> other    (no matching training class; organizer: map to other)
    CLASS_MAP_OVERRIDES = {
        'CMP': 'scratch',
        'VIA': 'other',
    }

    # Normalization — ImageNet mean/std (from model_metadata.json)
    # Applied AFTER PIL ToTensor (scales [0,255] to [0,1])
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD  = [0.229, 0.224, 0.225]

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

log_path = os.path.join(Config.OUTPUT_DIR, 'prediction_log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PREPROCESSING — matches training EXACTLY
#
# Training used (transfer_train.py / Phase 6 evaluation):
#   transforms.Grayscale(num_output_channels=1)
#   transforms.Lambda(lambda x: x.convert('RGB'))
#   transforms.Resize(IMAGE_SIZE)
#   transforms.ToTensor()
#   transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#
# PIL-based (NOT cv2) — avoids BGR/RGB channel mismatch
# ============================================================================

class GrayscaleToRGB:
    """Convert image to grayscale then back to RGB (3 identical channels).
    Avoids Lambda serialization issues on Windows multiprocessing."""
    def __call__(self, img):
        return img.convert('L').convert('RGB')


def get_transform():
    return transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize(Config.IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
    ])

# ============================================================================
# DATASET
# ============================================================================

class HackathonTestDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir  = root_dir
        self.transform = get_transform()
        self.samples   = []   # (image_path, test_class_name)

        for class_name in Config.TEST_CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                logger.warning(f"  Folder not found, skipping: {class_name}")
                continue

            imgs_found = 0
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.samples.append((os.path.join(class_dir, fname), class_name))
                    imgs_found += 1

            logger.info(f"  Loaded {imgs_found:3d} images from: {class_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        try:
            image = Image.open(img_path)
            tensor = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # Return blank tensor on error
            tensor = get_transform()(Image.new('RGB', Config.IMG_SIZE))
        return tensor, class_name, img_path

# ============================================================================
# CLASS MAPPING
# ============================================================================

def build_class_mapping():
    """Map each test folder name to a training class index."""
    mapping = {}

    for test_class in Config.TEST_CLASSES:

        # Override first (CMP, VIA)
        if test_class in Config.CLASS_MAP_OVERRIDES:
            target = Config.CLASS_MAP_OVERRIDES[test_class]
            idx    = Config.TRAINING_CLASSES.index(target)
            mapping[test_class] = idx
            logger.info(f"  {test_class:<10} -> {target:<10} (override, class {idx})")
            continue

        # Case-insensitive match
        matched = False
        for i, train_cls in enumerate(Config.TRAINING_CLASSES):
            if test_class.lower() == train_cls.lower():
                mapping[test_class] = i
                logger.info(f"  {test_class:<10} -> {train_cls:<10} (matched, class {i})")
                matched = True
                break

        if not matched:
            # Fallback to other
            idx = Config.TRAINING_CLASSES.index('other')
            mapping[test_class] = idx
            logger.warning(f"  {test_class:<10} -> other      (no match, class {idx})")

    return mapping

# ============================================================================
# ONNX INFERENCE
#
# Key points (Parth's checklist):
#   Step 1: Class order hardcoded — NOT from os.listdir()
#   Step 2: Input shape (1,3,224,224) — PIL ToTensor gives CHW automatically
#   Step 3: Normalization applied via torchvision transforms (same as training)
#   Step 4: Plain argmax on logits — no thresholding
#   Step 5: 5-image debug printed before full inference
# ============================================================================

def load_onnx_session(model_path):
    """Load ONNX model with onnxruntime."""
    logger.info(f"Loading ONNX model: {model_path}")

    # Check .data file exists alongside .onnx (external data format)
    data_file = model_path + '.data'
    if os.path.exists(data_file):
        logger.info(f"Found external data file: {os.path.basename(data_file)}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    # Use CPU provider (matches NXP eIQ deployment)
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )

    # Verify input/output shapes
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    logger.info(f"ONNX input  name={inp.name}, shape={inp.shape}, dtype={inp.type}")
    logger.info(f"ONNX output name={out.name}, shape={out.shape}, dtype={out.type}")

    return session, inp.name


def debug_5_images(session, input_name, dataset):
    """Parth Step 5: debug predictions on first 5 images before full run."""
    logger.info("\n[DEBUG] First 5 images:")

    for i in range(min(5, len(dataset))):
        tensor, class_name, img_path = dataset[i]

        # tensor shape: (3, 224, 224) from ToTensor
        arr = tensor.numpy()                    # CHW float32
        arr = np.expand_dims(arr, axis=0)       # -> (1,3,224,224)

        logger.info(f"  img[{i}] shape={arr.shape}  "
                    f"dtype={arr.dtype}  "
                    f"min={arr.min():.3f}  max={arr.max():.3f}")

        logits = session.run(None, {input_name: arr})[0]  # (1,8)
        pred   = int(np.argmax(logits))
        conf   = float(np.max(np.exp(logits) / np.sum(np.exp(logits))))

        logger.info(f"         True={class_name:<10}  "
                    f"Pred={Config.TRAINING_CLASSES[pred]:<10}  "
                    f"Confidence={conf:.3f}")


def run_onnx_inference(session, input_name, test_loader):
    """Run full inference using ONNX runtime."""
    logger.info("\n" + "="*80)
    logger.info("RUNNING ONNX INFERENCE")
    logger.info("="*80)

    all_predictions   = []
    all_probabilities = []
    all_true_labels   = []
    all_file_paths    = []

    for images, class_names, file_paths in tqdm(test_loader, desc="ONNX Inference"):
        # images: (batch, 3, 224, 224) torch tensor from DataLoader
        arr = images.numpy()  # -> numpy (batch, 3, 224, 224) float32

        # ONNX inference
        logits = session.run(None, {input_name: arr})[0]   # (batch, 8)

        # Softmax probabilities
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs      = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Argmax — plain, no thresholding (Parth Step 4)
        preds = np.argmax(logits, axis=1)

        all_predictions.extend(preds.tolist())
        all_probabilities.extend(probs.tolist())
        all_true_labels.extend(class_names)
        all_file_paths.extend(file_paths)

    logger.info(f"[SUCCESS] ONNX inference complete: {len(all_predictions)} predictions")
    return all_predictions, all_probabilities, all_true_labels, all_file_paths

# ============================================================================
# RESULTS & METRICS
# ============================================================================

def save_results(predictions, probabilities, true_labels, file_paths, class_mapping):

    true_indices = [class_mapping[lbl] for lbl in true_labels]

    # --- Metrics ---
    accuracy           = accuracy_score(true_indices, predictions)
    precision_macro    = precision_score(true_indices, predictions, average='macro',    zero_division=0)
    recall_macro       = recall_score(true_indices, predictions,    average='macro',    zero_division=0)
    f1_macro           = f1_score(true_indices, predictions,        average='macro',    zero_division=0)
    precision_weighted = precision_score(true_indices, predictions, average='weighted', zero_division=0)
    recall_weighted    = recall_score(true_indices, predictions,    average='weighted', zero_division=0)
    f1_weighted        = f1_score(true_indices, predictions,        average='weighted', zero_division=0)

    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Overall Accuracy:      {accuracy*100:.2f}%")
    logger.info(f"Macro  - P: {precision_macro*100:.2f}%  "
                f"R: {recall_macro*100:.2f}%  "
                f"F1: {f1_macro*100:.2f}%")
    logger.info(f"Weighted- P: {precision_weighted*100:.2f}%  "
                f"R: {recall_weighted*100:.2f}%  "
                f"F1: {f1_weighted*100:.2f}%")
    logger.info("="*80)

    # --- Classification report ---
    report = classification_report(
        true_indices, predictions,
        labels=list(range(len(Config.TRAINING_CLASSES))),
        target_names=Config.TRAINING_CLASSES,
        zero_division=0,
        digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")

    with open(os.path.join(Config.OUTPUT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write("Phase 2 Final Classification Report\n")
        f.write("Model: MobileNetV2 ONNX (Phase 1 submitted model)\n")
        f.write("Preprocessing: Grayscale->RGB->Resize 224x224->Normalize (per training)\n")
        f.write("Rules: Resize only, no TTA (per organizer email Feb 16 2026)\n")
        f.write("CMP->scratch (organizer confirmed), VIA->other\n")
        f.write("="*80 + "\n\n")
        f.write(report)

    # --- Predictions CSV ---
    pd.DataFrame({
        'file_path':       file_paths,
        'true_class':      true_labels,
        'true_index':      true_indices,
        'predicted_class': [Config.TRAINING_CLASSES[p] for p in predictions],
        'predicted_index': predictions,
        'confidence':      [max(prob) for prob in probabilities],
    }).to_csv(os.path.join(Config.OUTPUT_DIR, 'predictions_detailed.csv'), index=False)

    # --- Confusion matrix ---
    cm = confusion_matrix(
        true_indices, predictions,
        labels=list(range(len(Config.TRAINING_CLASSES)))
    )

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.TRAINING_CLASSES,
                yticklabels=Config.TRAINING_CLASSES,
                ax=axes[0])
    axes[0].set_title(
        f'Confusion Matrix (Counts)\n'
        f'Accuracy: {accuracy*100:.2f}% | Macro F1: {f1_macro*100:.2f}%',
        fontsize=13, pad=12
    )
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)

    # Normalized
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=Config.TRAINING_CLASSES,
                yticklabels=Config.TRAINING_CLASSES,
                ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=13, pad=12)
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # --- Metrics JSON ---
    metrics_out = {
        'timestamp':            datetime.now().isoformat(),
        'model':                'MobileNetV2 ONNX (Phase 1 submitted)',
        'onnx_model':           os.path.basename(Config.ONNX_MODEL_PATH),
        'preprocessing':        'Grayscale->RGB->Resize 224x224 BICUBIC->Normalize (PIL, matches training)',
        'tta_enabled':          False,
        'rules_note':           'Resize only per organizer email Feb 16 2026',
        'class_mapping_notes':  'CMP->scratch (organizer confirmed), VIA->other',
        'accuracy':             round(float(accuracy) * 100, 2),
        'precision_macro':      round(float(precision_macro) * 100, 2),
        'recall_macro':         round(float(recall_macro) * 100, 2),
        'f1_macro':             round(float(f1_macro) * 100, 2),
        'precision_weighted':   round(float(precision_weighted) * 100, 2),
        'recall_weighted':      round(float(recall_weighted) * 100, 2),
        'f1_weighted':          round(float(f1_weighted) * 100, 2),
        'total_images':         len(predictions),
        'confusion_matrix':     cm.tolist(),
        'training_classes':     Config.TRAINING_CLASSES,
        'phase1_test_accuracy': 92.87,
        'phase2_test_accuracy': round(float(accuracy) * 100, 2),
    }

    with open(os.path.join(Config.OUTPUT_DIR, 'metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_out, f, indent=2)

    logger.info(f"\n[OK] predictions_detailed.csv")
    logger.info(f"[OK] classification_report.txt")
    logger.info(f"[OK] confusion_matrix.png")
    logger.info(f"[OK] metrics_summary.json")
    logger.info(f"[OK] prediction_log.txt")

    return accuracy, f1_macro

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "="*80)
    logger.info("PHASE 2 - FINAL SUBMISSION (ONNX)")
    logger.info("Model: defect_classification_model.onnx (Phase 1 submitted)")
    logger.info("Preprocessing: Grayscale->RGB->Resize 224x224->Normalize (PIL)")
    logger.info("Rules: Resize only, no TTA (per organizer email Feb 16 2026)")
    logger.info("="*80)

    # --- [1] Load ONNX model ---
    logger.info("\n[1/5] Loading ONNX model...")
    session, input_name = load_onnx_session(Config.ONNX_MODEL_PATH)
    logger.info("[SUCCESS] ONNX model loaded")

    # --- [2] Build class mapping ---
    logger.info("\n[2/5] Building class mapping...")
    class_mapping = build_class_mapping()

    # --- [3] Load dataset ---
    logger.info("\n[3/5] Loading dataset...")
    test_dataset = HackathonTestDataset(Config.TEST_DATA_DIR)
    test_loader  = DataLoader(
        test_dataset,
        batch_size  = Config.BATCH_SIZE,
        shuffle     = False,
        num_workers = Config.NUM_WORKERS,
        drop_last   = False,
    )
    logger.info(f"[SUCCESS] Dataset loaded: {len(test_dataset)} images")

    # --- [4] Debug 5 images (Parth Step 5) ---
    logger.info("\n[4/5] Running debug check on 5 images...")
    debug_5_images(session, input_name, test_dataset)

    # --- [5] Full inference ---
    logger.info("\n[5/5] Running full ONNX inference...")
    predictions, probabilities, true_labels, file_paths = run_onnx_inference(
        session, input_name, test_loader
    )

    # --- Save & display results ---
    accuracy, f1 = save_results(
        predictions, probabilities, true_labels, file_paths, class_mapping
    )

    logger.info(f"\n[SUCCESS] All results saved to: {Config.OUTPUT_DIR}")
    logger.info("="*80)
    logger.info("PHASE 2 COMPLETE")
    logger.info(f"   Accuracy : {accuracy*100:.2f}%")
    logger.info(f"   F1 Macro : {f1*100:.2f}%")
    logger.info("="*80)


if __name__ == "__main__":

    main()
