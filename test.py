from ppe_framework import PPEDetector
import cv2

# Initialize detector
detector = PPEDetector(
    person_model_path=r"C:\Users\phakin.thongla-ar\OneDrive - Autoliv\Desktop\Safety\Dashboard\API\new_models\person.pt",
    ppe_model_path=r"C:\Users\phakin.thongla-ar\OneDrive - Autoliv\Desktop\Safety\Dashboard\API\new_models\object_train7.pt",
    classification_model_path=r"C:\Users\phakin.thongla-ar\OneDrive - Autoliv\Desktop\Safety\Dashboard\API\new_models\classify_trin9.pt",
    device='cpu',
    confidence_threshold=0.7,
    use_half_precision=True
)

# Load image
frame = cv2.imread("PPE55.jpg")

# Run detection
result = detector.detect(frame)

# Check results
print(f"Detected: {len(result['detections'])} items")
print(f"Non-safety items: {result['ng_count']}")
print(f"Has violations: {result['has_ng']}")

# Save annotated image
cv2.imwrite("result.jpg", result['annotated_frame'])