Phase-2 — Semiconductor Defect Classification (ONNX Inference)

This script runs the official Phase-2 evaluation using the same MobileNetV2 ONNX model submitted in Phase-1.

✔ Rules followed
No retraining, no re-export
Resize-only preprocessing (224×224)
No TTA or image enhancement
ONNX Runtime CPU (NXP eIQ compatible)

🧠 Pipeline
Grayscale → RGB → Resize 224×224 → Normalize → ONNX inference → Metrics
🔁 Class Mapping
CMP → scratch (organiser confirmed)
VIA → other (no training class)
📊 Outputs
Predictions CSV
Classification report
Confusion matrix
Metrics JSON
