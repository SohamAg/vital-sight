import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

class VitalSightDetector:
    def __init__(self, model_path="yolov8n.pt", device=None, buffer_len=30):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"[INFO] Model loaded on {self.device}")

        # Ring buffer for motion / state tracking per person
        self.buffer = deque(maxlen=buffer_len)
        self.last_detection_time = 0
        self.cooldown = 3  # seconds between event logs

    def _calculate_motion(self, boxes):
        """Compute rough motion intensity using bounding box displacement."""
        if len(self.buffer) < 2:
            return 0
        prev = self.buffer[-2]
        curr = self.buffer[-1]
        if len(prev) == 0 or len(curr) == 0:
            return 0

        deltas = []
        for pbox in prev:
            for cbox in curr:
                deltas.append(np.linalg.norm(np.array(pbox[:2]) - np.array(cbox[:2])))
        return np.mean(deltas) if deltas else 0

    def infer_frame(self, frame):
        """Run YOLO detection on a single frame."""
        results = self.model.predict(frame, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes.xywh.cpu().numpy():
                boxes.append(box)
        self.buffer.append(boxes)
        return results

    def classify_event(self, frame, motion_value):
        """Simple heuristic for the 5 categories."""
        h, w, _ = frame.shape
        event = None
        confidence = 0.0

        # Fire detection via pixel color heuristic
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fire_mask = cv2.inRange(hsv, (0, 50, 200), (50, 255, 255))
        fire_pixels = np.sum(fire_mask > 0)
        fire_ratio = fire_pixels / (h * w)

        # Motion thresholds
        if fire_ratio > 0.02:
            event, confidence = "fire", fire_ratio
        elif motion_value > 25:
            event, confidence = "violence_panic", motion_value / 100
        elif 5 < motion_value < 20:
            event, confidence = "fall", motion_value / 50
        elif motion_value <= 2:
            event, confidence = "immobility", 0.8
        else:
            event, confidence = "respiratory_distress", 0.6

        return event, confidence

    def process_video(self, source=0, display=True, save_output=False):
        """Main inference loop for video/webcam."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[ERROR] Cannot open source:", source)
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if save_output:
            out = cv2.VideoWriter("output_detected.mp4", fourcc, fps,
                                  (int(cap.get(3)), int(cap.get(4))))

        print("[INFO] Starting stream... Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.infer_frame(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            motion_value = self._calculate_motion(self.buffer)
            event, conf = self.classify_event(frame, motion_value)

            if event and (time.time() - self.last_detection_time > self.cooldown):
                print(f"[EVENT] {event.upper()} detected with conf {conf:.2f}")
                self.last_detection_time = time.time()

            # Visualization
            annotated = results[0].plot()
            cv2.putText(annotated, f"{event or 'none'} ({conf:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if event else (0, 255, 0), 2)
            if display:
                cv2.imshow("VitalSight Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_output and out:
                out.write(annotated)

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("[INFO] Stream ended.")
