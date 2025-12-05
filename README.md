# PPE Detection Framework

A comprehensive Python framework for detecting and classifying Personal Protective Equipment (PPE) using YOLO models with 3-stage detection pipeline.

## ✨ Features

- 🎯 **3-Stage Detection**: Person Detection → PPE Detection → Safety Classification
- 🚀 **GPU Accelerated**: CUDA support with FP16 precision
- 📍 **ROI & Exclusion Zones**: Define specific areas for detection
- 🔌 **Easy Integration**: Simple API for image, video, and real-time streams
- 📦 **Configurable**: Flexible configuration system
- 🎨 **Annotated Output**: Visual bounding boxes with classifications

## 🔧 Installation

### From Source

```bash
git clone https://github.com/nnackpt/ppe-detection-framework
cd ppe-detection-framework
pip install -e .
```

### From PyPI (if published)

```bash
pip install ppe-detection-framework
```

## 📋 Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended)
- YOLO models (person, PPE, classification)

## 🚀 Quick Start

### 1. Basic Usage

```python
from ppe_framework import PPEDetector
import cv2

# Initialize detector
detector = PPEDetector(
    person_model_path="models/person.pt",
    ppe_model_path="models/ppe.pt",
    classification_model_path="models/classify.pt",
    device='cuda',
    confidence_threshold=0.7
)

# Load image
frame = cv2.imread("test.jpg")

# Run detection
result = detector.detect(frame)

# Check results
print(f"Detected: {len(result['detections'])} items")
print(f"Non-safety items: {result['ng_count']}")
print(f"Has violations: {result['has_ng']}")

# Save annotated image
cv2.imwrite("result.jpg", result['annotated_frame'])
```

### 2. With ROI Zones

```python
# Define detection zones
roi_zones = [(100, 100), (500, 100), (500, 400), (100, 400)]
exclusion_zones = [(200, 200), (300, 200), (300, 300), (200, 300)]

result = detector.detect(
    frame,
    roi_zones=roi_zones,
    exclusion_zones=exclusion_zones
)
```

### 3. Real-time Video Stream

```python
import cv2

cap = cv2.VideoCapture(0)  # or RTSP URL

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = detector.detect(frame)
    
    cv2.imshow('PPE Detection', result['annotated_frame'])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### 4. FastAPI Integration

```python
from fastapi import FastAPI, File, UploadFile
from ppe_framework import PPEDetector

app = FastAPI()
detector = PPEDetector(
    person_model_path="models/person.pt",
    ppe_model_path="models/ppe.pt",
    device='cuda'
)

@app.post("/detect")
async def detect_ppe(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    result = detector.detect(frame)
    
    return {
        "detections": result['detections'],
        "ng_count": result['ng_count'],
        "has_ng": result['has_ng']
    }
```

## 📊 Detection Output

The `detect()` method returns a dictionary with:

```python
{
    'detections': [
        {
            'class': 'hand',
            'classified_as': 'non-safety-glove',
            'detection_conf': 0.85,
            'classification_conf': 0.92,
            'bbox': [x1, y1, x2, y2],
            'person_bbox': [px1, py1, px2, py2],
            'is_ng': True
        },
        # ... more detections
    ],
    'annotated_frame': numpy_array,  # Frame with bounding boxes
    'has_ng': True,                  # Boolean flag
    'ng_count': 2                    # Number of violations
}
```

## ⚙️ Configuration

### Using Config Classes

```python
from ppe_framework.config import ModelConfig, PPEConfig

# Model configuration
model_config = ModelConfig(
    person_model_path="models/person.pt",
    ppe_model_path="models/ppe.pt",
    classification_model_path="models/classify.pt",
    device='cuda',
    confidence_threshold=0.7,
    classification_threshold=0.6,
    use_half_precision=True
)

# Full configuration
config = PPEConfig(
    model=model_config,
    ng_save_dir="ng_images",
    ng_cooldown=5,
    save_original=True,
    save_annotated=True
)
```

### From Environment Variables

```python
from ppe_framework.config import PPEConfig

config = PPEConfig.from_env()
```

## 🎯 Supported PPE Types

- 👐 **Hand Protection**: Safety gloves vs non-safety gloves
- 👞 **Foot Protection**: Safety shoes
- 👓 **Eye Protection**: Safety glasses vs non-safety glasses
- 👕 **Body Protection**: Safety shirts vs non-safety clothing

## 📈 Performance

- **Inference Speed**: ~30-50ms per frame (NVIDIA RTX 3080)
- **Accuracy**: >90% for PPE classification
- **GPU Memory**: ~2-4GB VRAM

## 🔧 Advanced Usage

### Custom Model Paths

```python
detector = PPEDetector(
    person_model_path="custom/person_v2.pt",
    ppe_model_path="custom/ppe_v3.pt",
    classification_model_path="custom/classify_v2.pt"
)
```

### Batch Processing

```python
import glob

image_files = glob.glob("images/*.jpg")

for img_path in image_files:
    frame = cv2.imread(img_path)
    result = detector.detect(frame)
    # Process results...
```

## 📚 Examples

See the `examples/` directory for more usage examples:

- `basic_usage.py` - Simple image detection
- `video_stream.py` - Real-time video processing
- `batch_processing.py` - Process multiple images
- `fastapi_integration.py` - Web API example

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

For questions or support, please contact: koonnack55@gmail.com

## 🙏 Acknowledgments

- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV for image processing
- PyTorch for deep learning

---

Made with ❤️ for workplace safety