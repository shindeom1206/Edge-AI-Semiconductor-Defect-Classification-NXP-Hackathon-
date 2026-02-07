"""
Phase 6: Model Evaluation - FIXED for RGB input
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import sys

# ==================== CONFIGURATION ====================
MODEL_PATH = r"C:\hackathon_project\models\best_model.pth"
TEST_DATA_PATH = r"C:\edge-ai-defect-classification\dataset_128\test"
RESULTS_DIR = r"C:\hackathon_project\results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Device: {device}")

# ==================== LOAD MODEL ====================
print("\n" + "="*80)
print("📦 LOADING MODEL")
print("="*80)

sys.path.append(r"C:\hackathon_project\scripts")
from transfer_model import MobileNetV2Transfer

checkpoint = torch.load(MODEL_PATH, map_location=device)
config = checkpoint['config']

NUM_CLASSES = config['num_classes']
IMAGE_SIZE = config['image_size']
CLASS_NAMES = config['class_names']

print(f"✅ Model metadata:")
print(f"   Image size: {IMAGE_SIZE}")
print(f"   Number of classes: {NUM_CLASSES}")
print(f"   Class names: {CLASS_NAMES}")
print(f"   Best Val F1: {checkpoint['best_val_f1']*100:.2f}%")

# Create model and load weights
model = MobileNetV2Transfer(pretrained=False, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ Model loaded successfully!")

# ==================== LOAD TEST DATASET ====================
print("\n" + "="*80)
print("📊 LOADING TEST DATASET")
print("="*80)

# FIXED: Convert grayscale to RGB (3 channels) to match training
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Load as grayscale
    transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to RGB (duplicate channels)
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

test_dataset = datasets.ImageFolder(TEST_DATA_PATH, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

dataset_class_names = test_dataset.classes
print(f"✅ Test dataset loaded")
print(f"   Total test samples: {len(test_dataset)}")
print(f"   Classes: {dataset_class_names}")

# Count samples per class
class_counts = {}
for _, label in test_dataset.samples:
    class_name = test_dataset.classes[label]
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

print(f"\n📊 Test Set Distribution:")
for cls in dataset_class_names:
    count = class_counts.get(cls, 0)
    print(f"   {cls:12s}: {count:3d} images")

# ==================== GENERATE PREDICTIONS ====================
print("\n" + "="*80)
print("🔮 GENERATING PREDICTIONS")
print("="*80)

y_true = []
y_pred = []
y_pred_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        y_pred_probs.extend(probs.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_probs = np.array(y_pred_probs)

print(f"✅ Predictions complete: {len(y_true)} test samples")

# ==================== CLASSIFICATION REPORT ====================
print("\n" + "="*80)
print("📊 DETAILED CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_true, y_pred, target_names=dataset_class_names, digits=3))

# ==================== CONFUSION MATRIX ====================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=dataset_class_names, yticklabels=dataset_class_names,
            cbar_kws={'label': 'Number of Predictions'},
            linewidths=0.5, linecolor='gray', annot_kws={'size': 11})
plt.title('Confusion Matrix - Phase 5 Balanced Model\n(NXP Hackathon: Edge-AI Defect Classification)', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "phase6_confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Confusion matrix saved: {cm_path}")
plt.close()

# ==================== PER-CLASS ANALYSIS ====================
print("\n" + "="*80)
print("🎯 PER-CLASS PERFORMANCE BREAKDOWN (CRITICAL EVALUATION)")
print("="*80)
print(f"{'Status':<8} | {'Class':<12} | {'Accuracy':<10} | {'Correct/Total':<15} | {'Notes'}")
print("-"*80)

per_class_acc = cm.diagonal() / cm.sum(axis=1)
failing_classes = []
warning_classes = []

for i, class_name in enumerate(dataset_class_names):
    acc = per_class_acc[i] * 100
    correct = cm[i, i]
    total = cm.sum(axis=1)[i]
    
    if acc >= 75:
        status = "✅ PASS"
        notes = "Excellent"
    elif acc >= 70:
        status = "⚠️  WARN"
        notes = "Acceptable, monitor"
        warning_classes.append(class_name)
    elif acc >= 60:
        status = "❌ FAIL"
        notes = "Needs improvement"
        failing_classes.append(class_name)
    else:
        status = "❌❌ CRIT"
        notes = "Critical issue!"
        failing_classes.append(class_name)
    
    print(f"{status:<8} | {class_name:<12} | {acc:5.1f}%     | {correct:3d} / {total:3d}       | {notes}")

print("="*80)

# ==================== OVERALL SUMMARY ====================
overall_acc = np.mean(y_true == y_pred) * 100
classes_passing = np.sum(per_class_acc >= 0.75)
classes_70plus = np.sum(per_class_acc >= 0.70)

macro_precision = precision_score(y_true, y_pred, average='macro') * 100
macro_recall = recall_score(y_true, y_pred, average='macro') * 100
macro_f1 = f1_score(y_true, y_pred, average='macro') * 100

print(f"\n📈 OVERALL SUMMARY:")
print(f"   Total Test Samples:    {len(y_true)}")
print(f"   Overall Accuracy:      {overall_acc:.2f}%")
print(f"   Classes ≥75% accuracy: {classes_passing}/{NUM_CLASSES}")
print(f"   Classes ≥70% accuracy: {classes_70plus}/{NUM_CLASSES}")
print(f"\n   Macro-Avg Precision:   {macro_precision:.2f}%")
print(f"   Macro-Avg Recall:      {macro_recall:.2f}%")
print(f"   Macro-Avg F1-Score:    {macro_f1:.2f}%")

# ==================== DECISION ====================
print("\n" + "="*80)
print("🚦 DECISION & NEXT STEPS:")
print("="*80)

if classes_passing >= 7 and overall_acc >= 82:
    verdict = "EXCELLENT"
    action = "PROCEED IMMEDIATELY TO PHASE 7"
    quality = "HIGH"
    print(f"✅✅ VERDICT: {verdict}! Model is ready for deployment!")
    print(f"\n📋 RECOMMENDED ACTION:")
    print(f"   → {action} (Quantization)")
    print(f"   → Your model meets all hackathon requirements")
    print(f"   → Expected deliverable quality: {quality}")
    
elif classes_passing >= 6 and overall_acc >= 80:
    verdict = "GOOD"
    action = "PROCEED TO PHASE 7"
    quality = "MEDIUM-HIGH"
    print(f"⚠️✅ VERDICT: {verdict}! Model is acceptable with minor weaknesses")
    print(f"\n📋 RECOMMENDED ACTION:")
    print(f"   → {action} (Quantization)")
    print(f"   → Document weak classes in final report")
    if warning_classes or failing_classes:
        print(f"   → Weak classes: {', '.join(warning_classes + failing_classes)}")
    print(f"   → Expected deliverable quality: {quality}")
    
elif classes_passing >= 5 and overall_acc >= 75:
    verdict = "BORDERLINE"
    action = "CONSIDER ITERATION OR PROCEED"
    quality = "MEDIUM"
    print(f"⚠️⚠️ VERDICT: {verdict} - Consider iteration")
    print(f"\n📋 RECOMMENDED ACTION (Choose one):")
    print(f"   Option A: Accept results and proceed to Phase 7")
    print(f"            (Time-constrained approach)")
    print(f"   Option B: One quick iteration:")
    print(f"            - Increase class weights for weak classes")
    print(f"            - Train 10-15 more epochs with LR=1e-5")
    if failing_classes:
        print(f"   → Failing classes: {', '.join(failing_classes)}")
    print(f"   → Expected deliverable quality: {quality}")
    
else:
    verdict = "NEEDS IMPROVEMENT"
    action = "DO NOT PROCEED - FIX MODEL FIRST"
    quality = "LOW"
    print(f"❌ VERDICT: {verdict} - Model not ready")
    print(f"\n📋 REQUIRED ACTION:")
    print(f"   → {action}")
    print(f"   → Address failing classes first")
    if failing_classes:
        print(f"   → Failing classes: {', '.join(failing_classes)}")

print("="*80)

# ==================== SAVE RESULTS ====================
results_file = os.path.join(RESULTS_DIR, "phase6_evaluation_report.txt")
with open(results_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("PHASE 6 EVALUATION REPORT - Balanced Model Performance\n")
    f.write("NXP Hackathon: Edge-AI Semiconductor Defect Classification\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test Data: {TEST_DATA_PATH}\n")
    f.write(f"Total Samples: {len(y_true)}\n\n")
    
    f.write("PER-CLASS ACCURACY:\n")
    f.write("-"*80 + "\n")
    for i, class_name in enumerate(dataset_class_names):
        acc = per_class_acc[i] * 100
        f.write(f"{class_name:12s}: {acc:5.1f}% ({cm[i, i]}/{cm.sum(axis=1)[i]})\n")
    
    f.write(f"\nOVERALL METRICS:\n")
    f.write(f"Overall Accuracy: {overall_acc:.2f}%\n")
    f.write(f"Classes ≥75%: {classes_passing}/{NUM_CLASSES}\n")
    f.write(f"Macro Precision: {macro_precision:.2f}%\n")
    f.write(f"Macro Recall: {macro_recall:.2f}%\n")
    f.write(f"Macro F1-Score: {macro_f1:.2f}%\n")
    f.write(f"\nVERDICT: {verdict}\n")
    f.write(f"ACTION: {action}\n")
    f.write(f"QUALITY: {quality}\n")

print(f"\n✅ Evaluation report saved: {results_file}")

# Save detailed metrics JSON
import json
metrics_dict = {
    'overall_accuracy': float(overall_acc),
    'macro_precision': float(macro_precision),
    'macro_recall': float(macro_recall),
    'macro_f1': float(macro_f1),
    'per_class_accuracy': {class_name: float(per_class_acc[i]*100) 
                          for i, class_name in enumerate(dataset_class_names)},
    'confusion_matrix': cm.tolist(),
    'class_names': dataset_class_names,
    'verdict': verdict,
    'classes_passing': int(classes_passing)
}

metrics_json = os.path.join(RESULTS_DIR, "phase6_detailed_metrics.json")
with open(metrics_json, 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print(f"✅ Detailed metrics (JSON) saved: {metrics_json}")
print("\n🎬 Phase 6 Complete! Review results and proceed accordingly.")
print("\n" + "="*80)