"""
PPE Detection Framework - Core Detector Module
3-Stage Detection: Person -> PPE Detection -> PPE Classification
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

class PPEDetector:
    def __init__(
        self,
        person_model_path: str,
        ppe_model_path: str,
        classification_model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.7,
        classification_threshold: float = 0.6,
        use_half_precision: bool = True
    ):
        # Auto-detect device if cuda not available
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = 'cpu'
            
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.classification_threshold = classification_threshold
        self.use_half_precision = use_half_precision and device == 'cuda'
        
        if use_half_precision and device == 'cpu':
            print("⚠️  Half precision (FP16) only works on CUDA. Using FP32 on CPU.")
        
        # Load models
        self.person_model = self._load_model(person_model_path)
        self.ppe_model = self._load_model(ppe_model_path)
        self.classification_model = None
        
        if classification_model_path:
            self.classification_model = self._load_model(classification_model_path)
            
        # Class mapping for classification
        self.class_mapping = {
            "hand": ["non-safety-glove", "safety-glove"],
            "shoe": ["safety-shoe", "non-safety-shoe"],
            "glasses": ["non-safety-glasses", "safety-glasses"],
            "shirt": ["non-safety-shirt", "safety-shirt"]
        }
        
        print(f"PPE Detector initialized on {self.device}")
        if self.use_half_precision:
            print(f"Using FP16 precision for faster inference")
        
    def _load_model(self, model_path: str) -> YOLO:
        try:
            model = YOLO(model_path)
            model.to(self.device)
            
            if self.device == 'cuda' and self.use_half_precision:
                try:
                    model.model.half()
                except Exception as e:
                    print(f"Could not convert model to FP16: {e}")
                    print("Falling back to FP32")
                    self.use_half_precision = False
            
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def detect(
        self,
        frame: np.ndarray,
        roi_zones: Optional[List[Tuple[int, int]]] = None,
        exclusion_zones: Optional[List[Tuple[int, int]]] = None
    ) -> Dict:
        annotated_frame = frame.copy()
        detections = []
        has_ng = False
        
        # Stage 1: Detect persons
        person_results = self.person_model(
            frame,
            device=self.device,
            verbose=False,
            half=self.use_half_precision,
            imgsz=640
        )
        
        # Process each person
        for person_result in person_results:
            if len(person_result.boxes) == 0:
                continue
            
            for person_box in person_result.boxes:
                person_conf = float(person_box.conf)
                if person_conf < self.confidence_threshold:
                    continue
                
                person_bbox = person_box.xyxy[0].cpu().numpy()
                px1, py1, px2, py2 = map(int, person_bbox)
                
                # Check ROI
                center_x = int((px1 + px2) / 2)
                center_y = int((py1 + py2) / 2)
                
                if roi_zones and not self._is_point_in_polygon((center_x, center_y), roi_zones):
                    continue
                
                if exclusion_zones and self._is_point_in_polygon((center_x, center_y), roi_zones):
                    continue
                
                # Stage 2: Detect PPE in person bbox
                ppe_detections = self._detect_ppe_in_person(frame, person_bbox)
                
                # Stage 3: Classify each PPE
                for ppe in ppe_detections:
                    bbox = ppe["bbox"]
                    x1, y1, x2, y2 = bbox
                    class_name = ppe["class"]
                    det_conf = ppe["conf"]
                    
                    # Classify
                    classified_name, class_conf, is_classified = self._classify_object(
                        frame, bbox, class_name
                    )
                    
                    if not is_classified or class_conf < self.classification_threshold:
                        continue
                    
                    # Check if NG
                    is_ng = "non-safety" in classified_name.lower()
                    if is_ng:
                        has_ng = True
                        
                    detections.append({
                        "class": class_name,
                        "classified_as": classified_name,
                        "detection_conf": round(det_conf, 2),
                        "classification_conf": round(class_conf, 2),
                        "bbox": bbox,
                        "person_bbox": [px1, py1, px2, py2],
                        "is_ng": is_ng 
                    })
                    
                    # Draw on annotated frame
                    color = (0, 0, 255) if is_ng else (0, 255, 0)
                    thickness = 4 if is_ng else 2
                    cv2.rectangle(annotated_frame, (x1, y1,), (x2, y2), color, thickness)
                    
                    display_name = classified_name.split("_", 1)[1] if "_" in classified_name else classified_name
                    cv2.putText(annotated_frame, display_name, (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        ng_count = sum(1 for d in detections if d["is_ng"])
        
        return {
            "detections": detections,
            "annotated_frame": annotated_frame,
            "has_ng": has_ng,
            "ng_count": ng_count
        }
        
    def _detect_ppe_in_person(self, frame: np.ndarray, person_bbox: np.ndarray) -> List[Dict]:
        x1, y1, x2, y2 = map(int, person_bbox)
        
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return []
        
        # Run PPE Detection
        ppe_results = self.ppe_model(
            person_crop,
            device=self.device,
            verbose=False,
            half=self.use_half_precision
        )
        
        detections = []
        for result in ppe_results:
            if len(result.boxes) == 0:
                continue
            
            for box in result.boxes:
                conf = float(box.conf)
                if conf < self.confidence_threshold:
                    continue
                
                crop_x1, crop_y1, crop_x2, crop_y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to original frame coordinates
                orig_x1 = int(crop_x1 + x1)
                orig_y1 = int(crop_y1 + y1)
                orig_x2 = int(crop_x2 + x1)
                orig_y2 = int(crop_y2 + y1)
                
                cls = int(box.cls)
                class_name = self.ppe_model.names[cls]
                
                detections.append({
                    "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                    "conf": conf,
                    "class": class_name,
                    "cls": cls
                })
                
        return detections
    
    def _classify_object(self, frame: np.ndarray, bbox: List[int], detected_class: str) -> Tuple[str, float, bool]:
        if self.classification_model is None:
            return detected_class, 0.0, False
        
        base_class = detected_class.lower()
        if base_class not in self.class_mapping:
            return detected_class, 0.0, False
        
        try:
            x1, y1, x2, y2 = bbox
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return detected_class, 0.0, False
            
            results = self.classification_model(cropped, device=self.device, verbose=False)
            
            if len(results) > 0 and len(results[0].probs) > 0:
                probs = results[0].probs
                relevant_classes = self.class_mapping.get(base_class, [])
                
                best_conf = 0
                best_class = None
                
                for idx, conf in enumerate(probs.data):
                    class_name = self.classification_model.names[idx]
                    conf_value = float(conf)
                    
                    if class_name in relevant_classes and conf_value > best_conf:
                        best_conf = conf_value
                        best_class = class_name
                        
                if best_class is not None:
                    return best_class, best_conf, True
                
            return detected_class, 0.0, False
        
        except Exception as e:
            print(f"Classification error: {e}")
            return detected_class, 0.0, False
        
    @staticmethod
    def _is_point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
                p1x, p1y = p2x, p2y
                
        return inside