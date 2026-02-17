# Semiconductor Defect Classification Model - ONNX Deployment

## Model Information
- **Architecture**: MobileNetV2 (Transfer Learning)
- **Input Shape**: [batch_size, 3, 224, 224]
- **Output Classes**: 8
- **ONNX Opset**: 13
- **File Size**: 0.30 MB
- **Accuracy**: 99.00%

## Class Labels
0: LER, 1: bridge, 2: clean, 3: crack, 4: open, 5: other, 6: particle, 7: scratch

## Preprocessing Requirements
1. Convert grayscale images to RGB (duplicate channel)
2. Resize to 224x224
3. Normalize with ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

## Usage Example (Python + ONNX Runtime)
```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession('defect_classification_model.onnx')

# Preprocess image
image = Image.open('wafer.png').convert('L')  # Grayscale
image = image.convert('RGB')  # Convert to RGB
image = image.resize((224, 224))
image_array = np.array(image).astype(np.float32) / 255.0

# Normalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = (image_array - mean) / std

# Add batch dimension and transpose to NCHW
image_array = np.transpose(image_array, (2, 0, 1))
image_array = np.expand_dims(image_array, axis=0)

# Inference
outputs = session.run(None, {'input': image_array})
prediction = np.argmax(outputs[0])


## NXP eIQ Platform Deployment
1. Upload `defect_classification_model.onnx` to eIQ Toolkit
2. Follow NXP eIQ documentation for i.MX RT deployment
3. Configure preprocessing pipeline as described above
4. Expected inference time: ~4.5 ms per image (CPU)

## Performance
- **Inference Speed**: 4.48 ms per image
- **Throughput**: 223.3 images/second

## Files Included
- `defect_classification_model.onnx` - Main model file
- `model_metadata.json` - Model configuration and metadata
- `README.md` - This file

## Contact & Support
Hackathon Project: Edge-AI Semiconductor Defect Classification
Target Platform: NXP i.MX RT Series
