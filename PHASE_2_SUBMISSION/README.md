Phase-2 — Semiconductor Defect Classification (ONNX Inference)

This script runs the official Phase-2 evaluation using the same MobileNetV2 ONNX model submitted in Phase-1.

✔ Rules followed

1. No retraining,
2. no re-export
3. Resize-only preprocessing (224×224)
4. No TTA or image enhancement
5. ONNX Runtime CPU (NXP eIQ compatible)

🧠 Pipeline

Grayscale → RGB → Resize 224×224 → Normalize → ONNX inference → Metrics

🔁 Class Mapping

CMP → scratch (organiser confirmed)
VIA → other (no training class)

📊 Outputs

1. Predictions CSV
2. Classification report
3. Confusion matrix
4. Metrics JSON
