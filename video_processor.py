"""
Video Processing Pipeline for Emergency Detection
Handles webcam and video file input with YOLO-Pose integration
"""

import cv2
import numpy as np
from datetime import datetime
import os
from typing import Optional, List, Dict
from ultralytics import YOLO
import time

from config import (
    YOLO_MODEL, CONFIDENCE_THRESHOLD, VIDEO_CONFIG, 
    VISUALIZATION, KEYPOINT_DICT, SKELETON_CONNECTIONS
)
from emergency_detector import EmergencyDetector
from mcp_interface import MCPServerInterface
from vlm_placeholder import get_vlm_analyzer


class VideoProcessor:
    """
    Main video processing system for emergency detection
    Supports webcam and video file input
    """
    
    def __init__(self, source: Optional[str] = None):
        """
        Initialize video processor
        
        Args:
            source: Video source (None for webcam, path for video file)
        """
        self.source = source
        self.is_webcam = source is None or source.isdigit()
        
        # Initialize YOLO-Pose model
        print(f"[VideoProcessor] Loading YOLO-Pose model: {YOLO_MODEL}")
        self.yolo_model = YOLO(YOLO_MODEL)
        
        # Initialize VLM (disabled placeholder)
        self.vlm = get_vlm_analyzer()
        
        # Initialize emergency detector
        self.detector = EmergencyDetector()
        
        # Initialize MCP interface
        self.mcp = MCPServerInterface()
        
        # Alert management
        self.alert_cooldown = VIDEO_CONFIG['alert_cooldown']
        self.last_alerts = {}  # type -> timestamp
        self.alert_counter = 0
        
        # Video capture
        self.cap = None
        self.frame_count = 0
        self.fps = VIDEO_CONFIG['fps']
        
        # Ensure alert directory exists
        os.makedirs(VIDEO_CONFIG['alert_image_dir'], exist_ok=True)
        
        print(f"[VideoProcessor] Initialized successfully")
        print(f"[VideoProcessor] Source: {'Webcam' if self.is_webcam else source}")
    
    def start(self):
        """Start video processing"""
        # Open video source
        if self.is_webcam:
            self.cap = cv2.VideoCapture(0)
        else:
            if not os.path.exists(self.source):
                raise FileNotFoundError(f"Video file not found: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video source")
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) if not self.is_webcam else 30
        
        print(f"\n{'='*60}")
        print(f"VIDEO PROCESSING STARTED")
        print(f"{'='*60}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {self.fps}")
        print(f"Press 'q' to quit")
        print(f"{'='*60}\n")
        
        # Processing loop
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.is_webcam:
                        print("[VideoProcessor] Failed to read from webcam")
                        break
                    else:
                        print("[VideoProcessor] End of video file")
                        break
                
                self.frame_count += 1
                
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Handle detections
                if detections:
                    self._handle_detections(detections, processed_frame)
                
                # Display frame
                cv2.imshow('VitalSight Emergency Detection', processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[VideoProcessor] Quitting...")
                    break
        
        finally:
            self._cleanup()
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, detections)
        """
        # Run YOLO-Pose detection
        results = self.yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Run emergency detection (VLM is bypassed)
        detections = self.detector.process_frame(frame, results)
        
        # Visualize results
        visualized_frame = self._visualize_results(frame, results, detections)
        
        return visualized_frame, detections
    
    def _visualize_results(self, frame: np.ndarray, results, 
                          detections: List[Dict]) -> np.ndarray:
        """
        Visualize detection results on frame
        
        Args:
            frame: Input frame
            results: YOLO-Pose results
            detections: Emergency detections
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw pose detections
        if results and len(results) > 0:
            for result in results:
                if result.keypoints is None:
                    continue
                
                boxes = result.boxes
                keypoints = result.keypoints
                
                for i in range(len(boxes)):
                    # Draw bounding box
                    if VISUALIZATION['show_bounding_box']:
                        box = boxes[i].xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), 
                                    VISUALIZATION['text_color'], 2)
                    
                    # Get keypoints
                    kp = keypoints[i].data[0].cpu().numpy()
                    
                    # Draw skeleton connections
                    if VISUALIZATION['show_skeleton']:
                        for connection in SKELETON_CONNECTIONS:
                            pt1_idx, pt2_idx = connection
                            if (kp[pt1_idx, 2] > 0.5 and kp[pt2_idx, 2] > 0.5):
                                pt1 = (int(kp[pt1_idx, 0]), int(kp[pt1_idx, 1]))
                                pt2 = (int(kp[pt2_idx, 0]), int(kp[pt2_idx, 1]))
                                cv2.line(annotated, pt1, pt2, 
                                       VISUALIZATION['skeleton_color'], 2)
                    
                    # Draw keypoints
                    if VISUALIZATION['show_keypoints']:
                        for j in range(len(kp)):
                            if kp[j, 2] > 0.5:
                                x, y = int(kp[j, 0]), int(kp[j, 1])
                                cv2.circle(annotated, (x, y), 3, 
                                         VISUALIZATION['keypoint_color'], -1)
        
        # Draw detection alerts
        if detections:
            y_offset = 30
            for detection in detections:
                alert_text = f"ALERT: {detection['type']} (Conf: {detection['confidence']:.2f})"
                cv2.putText(annotated, alert_text, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                          VISUALIZATION['alert_color'], 2)
                y_offset += 35
        
        # Draw frame info
        info_text = f"Frame: {self.frame_count} | FPS: {self.fps}"
        cv2.putText(annotated, info_text, (10, annotated.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   VISUALIZATION['text_color'], 1)
        
        return annotated
    
    def _handle_detections(self, detections: List[Dict], frame: np.ndarray):
        """
        Handle detected emergencies
        
        Args:
            detections: List of detections
            frame: Current frame
        """
        current_time = time.time()
        
        for detection in detections:
            detection_type = detection['type']
            
            # Check cooldown
            if detection_type in self.last_alerts:
                time_since_last = current_time - self.last_alerts[detection_type]
                if time_since_last < self.alert_cooldown:
                    continue
            
            # Update last alert time
            self.last_alerts[detection_type] = current_time
            
            # Send MCP alert
            print(f"\n[ALERT] {detection_type} detected at frame {self.frame_count}")
            print(f"[ALERT] Confidence: {detection['confidence']:.2f}")
            print(f"[ALERT] Details: {detection.get('details', {})}")
            
            success = self.mcp.send_mcp_alert(
                action_type=detection_type,
                frame_id=self.frame_count,
                additional_info=detection.get('details', {})
            )
            
            if success:
                print(f"[ALERT] MCP notification sent successfully")
            
            # Save alert image
            if VIDEO_CONFIG['save_alert_images']:
                self._save_alert_image(frame, detection)
    
    def _save_alert_image(self, frame: np.ndarray, detection: Dict):
        """
        Save alert image to disk
        
        Args:
            frame: Frame to save
            detection: Detection information
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{self.alert_counter}_{timestamp}_{self.frame_count}.jpg"
        filepath = os.path.join(VIDEO_CONFIG['alert_image_dir'], filename)
        
        # Add alert text to image
        alert_frame = frame.copy()
        text = f"{detection['type']} - Frame {self.frame_count}"
        cv2.putText(alert_frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite(filepath, alert_frame)
        print(f"[ALERT] Saved alert image: {filepath}")
        
        self.alert_counter += 1
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total frames processed: {self.frame_count}")
        
        summary = self.detector.get_detection_summary()
        print(f"Total detections: {summary['total_detections']}")
        
        if summary['by_type']:
            print("\nDetections by type:")
            for det_type, count in summary['by_type'].items():
                print(f"  - {det_type}: {count}")
        
        alert_log = self.mcp.get_alert_log()
        print(f"\nTotal MCP alerts sent: {len(alert_log)}")
        print(f"Alert images saved: {self.alert_counter}")
        print(f"{'='*60}\n")


def main():
    """Main function for testing"""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        source = sys.argv[1]
        print(f"Processing video file: {source}")
    else:
        source = None
        print("Using webcam")
    
    try:
        processor = VideoProcessor(source)
        processor.start()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
