# ml_detector.py
import time
import cv2
import numpy as np
from ultralytics import YOLO

class ObjectPerceptionEngine:
    """Pre-trained YOLOv8 detector for object logging and obstacle identification."""
    
    def __init__(self, model_name="yolov8n.pt"):
        print(f"📦 Loading pre-trained perception model: {model_name}...")
        # Automatically downloads lightweight YOLOv8 Nano pre-trained on 80 COCO classes
        self.model = YOLO(model_name)
        print("✅ Perception Engine Ready.")

    def detect_and_log(self, frame):
        """Processes an image/frame, logs every detected object, and returns telemetry."""
        results = self.model(frame, verbose=False)[0]
        detected_objects = []

        timestamp = time.strftime("%H:%M:%S")

        if len(results.boxes) == 0:
            print(f"[{timestamp} LOG] Path Clear - No Obstacles Detected.")
            return [], frame

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            confidence = float(box.conf[0])
            bbox = [round(float(c), 1) for c in box.xyxy[0].tolist()] # [xmin, ymin, xmax, ymax]

            detection_entry = {
                "timestamp": timestamp,
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": bbox
            }
            detected_objects.append(detection_entry)

            # Structured Terminal Log Output
            print(f"[{timestamp} DETECTED] Class: '{label}' | Confidence: {confidence*100:.1f}% | BBox: {bbox}")

        annotated_frame = results.plot()
        return detected_objects, annotated_frame