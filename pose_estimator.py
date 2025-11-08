"""
Pose Estimation Module
Uses MediaPipe for real-time human pose estimation
"""

import cv2
import mediapipe as mp
import numpy as np
import config


class PoseEstimator:
    """Handles pose estimation using MediaPipe"""
    
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            enable_segmentation=False,
            smooth_landmarks=True
        )
        
        if config.DEBUG_MODE:
            print("✓ Pose estimator initialized")
    
    def process_frame(self, frame):
        """
        Process a frame and extract pose landmarks
        Returns: (landmarks, processed_frame)
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        return results
    
    def draw_landmarks(self, frame, results, color=None):
        """
        Draw pose landmarks on the frame
        """
        if not results.pose_landmarks:
            return frame
        
        # Use custom color if provided
        if color:
            landmark_drawing_spec = self.mp_drawing.DrawingSpec(
                color=color, thickness=2, circle_radius=2
            )
            connection_drawing_spec = self.mp_drawing.DrawingSpec(
                color=color, thickness=2
            )
        else:
            landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style()
            connection_drawing_spec = None
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )
        
        return frame
    
    def get_landmarks(self, results):
        """Extract landmarks from results"""
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None
    
    def get_bounding_box(self, landmarks, frame_width, frame_height):
        """
        Calculate bounding box around the person
        Returns: (x, y, w, h) or None
        """
        if not landmarks:
            return None
        
        # Get all landmark coordinates
        x_coords = [lm.x * frame_width for lm in landmarks]
        y_coords = [lm.y * frame_height for lm in landmarks]
        
        # Calculate bounding box
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame_width, x_max + padding)
        y_max = min(frame_height, y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def is_person_in_zone(self, landmarks, frame_width, frame_height, zones):
        """
        Check if person is in any of the detection zones
        Returns: True if in zone or no zones defined
        """
        if not zones or not landmarks:
            return True  # If no zones, monitor entire frame
        
        # Get center point of person (average of hip positions)
        try:
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            center_x = (left_hip.x + right_hip.x) / 2
            center_y = (left_hip.y + right_hip.y) / 2
        except (IndexError, AttributeError):
            return True  # If can't determine position, assume in zone
        
        # Check if center is in any zone
        for zone in zones:
            x1, y1, x2, y2 = zone
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                return True
        
        return False
    
    def draw_bounding_box(self, frame, bbox, color, label=None):
        """Draw bounding box on frame"""
        if not bbox:
            return frame
        
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        if label:
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def get_keypoint_visibility(self, landmarks):
        """
        Check visibility of key body parts
        Returns dict with visibility scores
        """
        if not landmarks:
            return None
        
        try:
            return {
                'nose': landmarks[0].visibility,
                'left_shoulder': landmarks[11].visibility,
                'right_shoulder': landmarks[12].visibility,
                'left_hip': landmarks[23].visibility,
                'right_hip': landmarks[24].visibility,
                'left_knee': landmarks[25].visibility,
                'right_knee': landmarks[26].visibility,
            }
        except (IndexError, AttributeError):
            return None
    
    def close(self):
        """Release resources"""
        self.pose.close()
        if config.DEBUG_MODE:
            print("✓ Pose estimator closed")
