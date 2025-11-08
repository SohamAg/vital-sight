"""
Distress Detection Module
Analyzes pose data to detect falls and distress signals
"""

import numpy as np
import time
from collections import deque
import config


class DistressDetector:
    """Detects distress signals from pose estimation data"""
    
    def __init__(self):
        self.distress_start_time = {}  # Track when distress started for each person
        self.position_history = {}     # Track position history for movement detection
        self.fall_detected = {}        # Track if fall was detected
        
    def calculate_body_angle(self, landmarks):
        """
        Calculate the angle of the body relative to vertical
        Returns angle in degrees (0 = vertical, 90 = horizontal)
        """
        if not landmarks:
            return None
            
        # Get key body points (using MediaPipe pose landmarks)
        # 11: left shoulder, 12: right shoulder
        # 23: left hip, 24: right hip
        try:
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
            left_hip = np.array([landmarks[23].x, landmarks[23].y])
            right_hip = np.array([landmarks[24].x, landmarks[24].y])
            
            # Calculate center points
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            
            # Calculate angle from vertical
            dx = hip_center[0] - shoulder_center[0]
            dy = hip_center[1] - shoulder_center[1]
            
            # Angle from vertical (0 degrees = standing, 90 degrees = horizontal)
            angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy))))
            
            return angle
        except (IndexError, AttributeError):
            return None
    
    def calculate_vertical_position(self, landmarks):
        """
        Calculate the vertical position (normalized) of the person's center of mass
        Lower values indicate person is closer to ground
        """
        if not landmarks:
            return None
            
        try:
            # Get hip position as approximation of center of mass
            left_hip = landmarks[23].y
            right_hip = landmarks[24].y
            center_y = (left_hip + right_hip) / 2
            
            return center_y
        except (IndexError, AttributeError):
            return None
    
    def detect_fall(self, landmarks, person_id=0):
        """
        Detect if a person has fallen based on pose analysis
        Returns: (is_distress, distress_type, confidence)
        """
        if not landmarks:
            return False, None, 0.0
        
        # Calculate body angle
        body_angle = self.calculate_body_angle(landmarks)
        if body_angle is None:
            return False, None, 0.0
        
        # Calculate vertical position
        vertical_pos = self.calculate_vertical_position(landmarks)
        if vertical_pos is None:
            return False, None, 0.0
        
        # Initialize tracking for this person if needed
        if person_id not in self.position_history:
            self.position_history[person_id] = deque(maxlen=config.TRACK_HISTORY_LENGTH)
            self.distress_start_time[person_id] = None
            self.fall_detected[person_id] = False
        
        # Store position history
        self.position_history[person_id].append({
            'angle': body_angle,
            'vertical_pos': vertical_pos,
            'timestamp': time.time()
        })
        
        # Detect fall (horizontal body orientation)
        is_horizontal = body_angle > config.FALL_DETECTION_THRESHOLD
        
        # Check if person is on the ground (high vertical position value)
        is_on_ground = vertical_pos > 0.6  # Bottom 40% of frame
        
        # Detect sudden vertical drop (fall in progress)
        rapid_fall = False
        if len(self.position_history[person_id]) > 5:
            recent_positions = list(self.position_history[person_id])[-5:]
            vertical_change = recent_positions[-1]['vertical_pos'] - recent_positions[0]['vertical_pos']
            time_diff = recent_positions[-1]['timestamp'] - recent_positions[0]['timestamp']
            
            if time_diff > 0:
                velocity = vertical_change / time_diff
                # Positive velocity = moving down in frame
                rapid_fall = velocity > 0.15  # Threshold for fall detection
        
        # Determine distress state
        is_distress = False
        distress_type = None
        confidence = 0.0
        
        if rapid_fall and not self.fall_detected[person_id]:
            # Active fall detected
            is_distress = True
            distress_type = "FALL_IN_PROGRESS"
            confidence = 0.9
            self.fall_detected[person_id] = True
            self.distress_start_time[person_id] = time.time()
            
        elif is_horizontal and is_on_ground:
            # Person lying on ground
            if self.distress_start_time[person_id] is None:
                self.distress_start_time[person_id] = time.time()
            
            # Check if person has been down for threshold time
            time_in_distress = time.time() - self.distress_start_time[person_id]
            
            if time_in_distress > config.DISTRESS_TIME_THRESHOLD:
                is_distress = True
                distress_type = "PERSON_DOWN"
                confidence = min(0.95, 0.6 + (time_in_distress / 60))  # Confidence increases with time
            else:
                # Person is down but not long enough yet
                distress_type = "POSSIBLE_DISTRESS"
                confidence = 0.4
        else:
            # Person appears normal
            self.distress_start_time[person_id] = None
            self.fall_detected[person_id] = False
        
        return is_distress, distress_type, confidence
    
    def detect_immobility(self, landmarks, person_id=0):
        """
        Detect if a person has been immobile for an extended period
        """
        if len(self.position_history.get(person_id, [])) < 10:
            return False, 0.0
        
        recent_positions = list(self.position_history[person_id])[-10:]
        
        # Calculate movement variance
        angles = [p['angle'] for p in recent_positions]
        vertical_positions = [p['vertical_pos'] for p in recent_positions]
        
        angle_variance = np.var(angles)
        position_variance = np.var(vertical_positions)
        
        # Low variance indicates immobility
        is_immobile = angle_variance < 5 and position_variance < 0.001
        
        if is_immobile and recent_positions[-1]['vertical_pos'] > 0.6:
            # Immobile and on ground
            time_immobile = recent_positions[-1]['timestamp'] - recent_positions[0]['timestamp']
            confidence = min(0.8, time_immobile / 60)
            return True, confidence
        
        return False, 0.0
    
    def get_distress_info(self, landmarks, person_id=0):
        """
        Get comprehensive distress information
        Returns dict with all distress indicators
        """
        is_fall, distress_type, fall_confidence = self.detect_fall(landmarks, person_id)
        is_immobile, immobile_confidence = self.detect_immobility(landmarks, person_id)
        
        body_angle = self.calculate_body_angle(landmarks)
        vertical_pos = self.calculate_vertical_position(landmarks)
        
        return {
            'is_distress': is_fall or is_immobile,
            'distress_type': distress_type if is_fall else ('IMMOBILE' if is_immobile else None),
            'confidence': max(fall_confidence, immobile_confidence),
            'body_angle': body_angle,
            'vertical_position': vertical_pos,
            'is_fall': is_fall,
            'is_immobile': is_immobile
        }
    
    def reset_person(self, person_id):
        """Reset tracking data for a specific person"""
        if person_id in self.position_history:
            del self.position_history[person_id]
        if person_id in self.distress_start_time:
            del self.distress_start_time[person_id]
        if person_id in self.fall_detected:
            del self.fall_detected[person_id]
