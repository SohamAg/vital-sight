"""
Emergency Action Detection Algorithms
Implements custom rule-based detection for 5 emergency scenarios using YOLO-Pose keypoints
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from config import DETECTION_PARAMS, KEYPOINT_DICT
import cv2


class PersonTracker:
    """Track individual person across frames"""
    
    def __init__(self, person_id: int, keypoints: np.ndarray, bbox: tuple):
        self.person_id = person_id
        self.keypoints_history = deque(maxlen=100)
        self.bbox_history = deque(maxlen=100)
        self.velocity_history = deque(maxlen=30)
        self.state = 'normal'
        self.state_duration = 0
        self.last_alert_frame = -1000
        
        self.keypoints_history.append(keypoints)
        self.bbox_history.append(bbox)
        
    def update(self, keypoints: np.ndarray, bbox: tuple):
        """Update tracker with new frame data"""
        self.keypoints_history.append(keypoints)
        self.bbox_history.append(bbox)
        
        # Calculate velocity
        if len(self.keypoints_history) >= 2:
            velocity = self._calculate_velocity()
            self.velocity_history.append(velocity)
    
    def _calculate_velocity(self) -> float:
        """Calculate keypoint velocity between last two frames"""
        if len(self.keypoints_history) < 2:
            return 0.0
        
        kp1 = self.keypoints_history[-2]
        kp2 = self.keypoints_history[-1]
        
        # Calculate average velocity across all valid keypoints
        velocities = []
        for i in range(len(kp1)):
            if kp1[i, 2] > 0.5 and kp2[i, 2] > 0.5:  # confidence threshold
                dx = kp2[i, 0] - kp1[i, 0]
                dy = kp2[i, 1] - kp1[i, 1]
                velocities.append(np.sqrt(dx**2 + dy**2))
        
        return np.mean(velocities) if velocities else 0.0
    
    def get_center(self) -> Tuple[float, float]:
        """Get current bounding box center"""
        if not self.bbox_history:
            return (0, 0)
        bbox = self.bbox_history[-1]
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def get_height(self) -> float:
        """Get current bounding box height"""
        if not self.bbox_history:
            return 0
        bbox = self.bbox_history[-1]
        return bbox[3] - bbox[1]


class EmergencyDetector:
    """
    Main emergency detection system using YOLO-Pose keypoints and custom algorithms
    """
    
    def __init__(self):
        """Initialize emergency detector"""
        self.trackers = {}
        self.next_person_id = 0
        self.frame_count = 0
        self.detections = []
        
        # Detection parameters
        self.violence_params = DETECTION_PARAMS['violence']
        self.fall_params = DETECTION_PARAMS['fall']
        self.fainting_params = DETECTION_PARAMS['fainting']
        self.cardiac_params = DETECTION_PARAMS['cardiac']
        self.fire_params = DETECTION_PARAMS['fire']
        
        # State tracking for complex detections
        self.cardiac_candidates = {}  # person_id -> frame_count
        self.fainting_candidates = {}  # person_id -> (fall_frame, immobility_count)
        
        print("[Detector] Emergency detector initialized")
        print("[Detector] VLM BYPASSED - Using YOLO-Pose + Custom Algorithms ONLY")
    
    def process_frame(self, frame: np.ndarray, pose_results) -> List[Dict]:
        """
        Process single frame and detect emergencies
        
        Args:
            frame: Input frame
            pose_results: YOLO-Pose detection results
            
        Returns:
            List of detected emergencies
        """
        self.frame_count += 1
        current_detections = []
        
        # Extract keypoints and bounding boxes from YOLO results
        persons = self._extract_persons(pose_results)
        
        # Update or create trackers
        self._update_trackers(persons)
        
        # Run detection algorithms
        violence_detected = self._detect_violence(persons)
        if violence_detected:
            current_detections.append(violence_detected)
        
        fall_detected = self._detect_sudden_fall(persons)
        if fall_detected:
            current_detections.append(fall_detected)
        
        fainting_detected = self._detect_fainting(persons)
        if fainting_detected:
            current_detections.append(fainting_detected)
        
        cardiac_detected = self._detect_cardiac_distress(persons, frame)
        if cardiac_detected:
            current_detections.append(cardiac_detected)
        
        fire_detected = self._detect_fire(frame)
        if fire_detected:
            current_detections.append(fire_detected)
        
        self.detections.extend(current_detections)
        return current_detections
    
    def _extract_persons(self, pose_results) -> List[Dict]:
        """Extract person data from YOLO-Pose results"""
        persons = []
        
        if pose_results is None or len(pose_results) == 0:
            return persons
        
        for result in pose_results:
            if result.keypoints is None:
                continue
                
            boxes = result.boxes
            keypoints = result.keypoints
            
            for i in range(len(boxes)):
                # Get bounding box
                box = boxes[i].xyxy[0].cpu().numpy()
                conf = boxes[i].conf[0].cpu().numpy()
                
                # Get keypoints (x, y, confidence)
                kp = keypoints[i].data[0].cpu().numpy()
                
                persons.append({
                    'bbox': box,
                    'confidence': float(conf),
                    'keypoints': kp,
                })
        
        return persons
    
    def _update_trackers(self, persons: List[Dict]):
        """Update person trackers"""
        # Simple nearest-neighbor tracking
        used_trackers = set()
        
        for person in persons:
            # Find closest tracker
            best_tracker_id = None
            best_distance = float('inf')
            
            person_center = (
                (person['bbox'][0] + person['bbox'][2]) / 2,
                (person['bbox'][1] + person['bbox'][3]) / 2
            )
            
            for tracker_id, tracker in self.trackers.items():
                if tracker_id in used_trackers:
                    continue
                
                tracker_center = tracker.get_center()
                distance = np.sqrt(
                    (person_center[0] - tracker_center[0])**2 + 
                    (person_center[1] - tracker_center[1])**2
                )
                
                if distance < best_distance and distance < 100:
                    best_distance = distance
                    best_tracker_id = tracker_id
            
            if best_tracker_id is not None:
                # Update existing tracker
                self.trackers[best_tracker_id].update(
                    person['keypoints'], 
                    person['bbox']
                )
                used_trackers.add(best_tracker_id)
            else:
                # Create new tracker
                new_tracker = PersonTracker(
                    self.next_person_id, 
                    person['keypoints'], 
                    person['bbox']
                )
                self.trackers[self.next_person_id] = new_tracker
                self.next_person_id += 1
    
    def _detect_violence(self, persons: List[Dict]) -> Optional[Dict]:
        """
        Detect violence/assault based on rapid aggressive movements and close proximity
        """
        if len(persons) < self.violence_params['min_subjects']:
            return None
        
        # Check for pairs of people with high velocity and close distance
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1 = persons[i]
                p2 = persons[j]
                
                # Calculate distance between centers
                center1 = ((p1['bbox'][0] + p1['bbox'][2]) / 2,
                          (p1['bbox'][1] + p1['bbox'][3]) / 2)
                center2 = ((p2['bbox'][0] + p2['bbox'][2]) / 2,
                          (p2['bbox'][1] + p2['bbox'][3]) / 2)
                
                distance = np.sqrt(
                    (center1[0] - center2[0])**2 + 
                    (center1[1] - center2[1])**2
                )
                
                # Check if close enough
                if distance > self.violence_params['distance_threshold']:
                    continue
                
                # Check velocities for both persons
                velocities = []
                for person_idx, person in enumerate([p1, p2]):
                    kp = person['keypoints']
                    # Calculate movement in key points (shoulders, elbows, hands)
                    key_indices = [5, 6, 7, 8, 9, 10]  # upper body
                    if len(self.trackers) > person_idx:
                        tracker_id = list(self.trackers.keys())[person_idx] if person_idx < len(self.trackers) else None
                        if tracker_id and tracker_id in self.trackers:
                            tracker = self.trackers[tracker_id]
                            if len(tracker.velocity_history) > 0:
                                velocities.append(tracker.velocity_history[-1])
                
                if len(velocities) >= 2:
                    avg_velocity = np.mean(velocities)
                    if avg_velocity > self.violence_params['velocity_threshold']:
                        return {
                            'type': 'Violence/Assault',
                            'confidence': 0.85,
                            'frame': self.frame_count,
                            'details': {
                                'num_subjects': len(persons),
                                'distance': float(distance),
                                'velocity': float(avg_velocity)
                            }
                        }
        
        return None
    
    def _detect_sudden_fall(self, persons: List[Dict]) -> Optional[Dict]:
        """
        Detect sudden falling based on sudden acceleration towards floor
        Focuses on rapid changes in downward movement across multiple keypoints
        """
        for person in persons:
            kp = person['keypoints']
            bbox = person['bbox']
            
            # Get key points for fall detection
            left_shoulder = kp[KEYPOINT_DICT['left_shoulder']]
            right_shoulder = kp[KEYPOINT_DICT['right_shoulder']]
            left_hip = kp[KEYPOINT_DICT['left_hip']]
            right_hip = kp[KEYPOINT_DICT['right_hip']]
            left_knee = kp[KEYPOINT_DICT['left_knee']]
            right_knee = kp[KEYPOINT_DICT['right_knee']]
            left_wrist = kp[KEYPOINT_DICT['left_wrist']]
            right_wrist = kp[KEYPOINT_DICT['right_wrist']]
            
            # Check if core keypoints are visible
            if (left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5 or 
                left_hip[2] < 0.5 or right_hip[2] < 0.5):
                continue
            
            # Calculate average positions for current frame
            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            
            # Include knees if visible
            avg_knee_y = None
            if left_knee[2] > 0.5 and right_knee[2] > 0.5:
                avg_knee_y = (left_knee[1] + right_knee[1]) / 2
            
            # Find corresponding tracker
            person_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            for tracker_id, tracker in self.trackers.items():
                tracker_center = tracker.get_center()
                distance = np.sqrt(
                    (person_center[0] - tracker_center[0])**2 + 
                    (person_center[1] - tracker_center[1])**2
                )
                
                # Need at least 3 frames to calculate acceleration
                if distance < 50 and len(tracker.keypoints_history) >= 3:
                    # Get previous 2 frames of keypoints
                    prev_kp_1 = tracker.keypoints_history[-2]  # t-1
                    prev_kp_2 = tracker.keypoints_history[-3]  # t-2
                    
                    prev_left_shoulder_1 = prev_kp_1[KEYPOINT_DICT['left_shoulder']]
                    prev_right_shoulder_1 = prev_kp_1[KEYPOINT_DICT['right_shoulder']]
                    prev_left_hip_1 = prev_kp_1[KEYPOINT_DICT['left_hip']]
                    prev_right_hip_1 = prev_kp_1[KEYPOINT_DICT['right_hip']]
                    
                    prev_left_shoulder_2 = prev_kp_2[KEYPOINT_DICT['left_shoulder']]
                    prev_right_shoulder_2 = prev_kp_2[KEYPOINT_DICT['right_shoulder']]
                    prev_left_hip_2 = prev_kp_2[KEYPOINT_DICT['left_hip']]
                    prev_right_hip_2 = prev_kp_2[KEYPOINT_DICT['right_hip']]
                    
                    # Check if previous keypoints are visible
                    if (prev_left_shoulder_1[2] < 0.5 or prev_right_shoulder_1[2] < 0.5 or 
                        prev_left_hip_1[2] < 0.5 or prev_right_hip_1[2] < 0.5 or
                        prev_left_shoulder_2[2] < 0.5 or prev_right_shoulder_2[2] < 0.5 or 
                        prev_left_hip_2[2] < 0.5 or prev_right_hip_2[2] < 0.5):
                        continue
                    
                    # Calculate positions for previous frames
                    prev_avg_shoulder_y_1 = (prev_left_shoulder_1[1] + prev_right_shoulder_1[1]) / 2
                    prev_avg_hip_y_1 = (prev_left_hip_1[1] + prev_right_hip_1[1]) / 2
                    
                    prev_avg_shoulder_y_2 = (prev_left_shoulder_2[1] + prev_right_shoulder_2[1]) / 2
                    prev_avg_hip_y_2 = (prev_left_hip_2[1] + prev_right_hip_2[1]) / 2
                    
                    # Calculate velocities (change in position)
                    # Current velocity (t to t-1)
                    shoulder_vel_current = avg_shoulder_y - prev_avg_shoulder_y_1
                    hip_vel_current = avg_hip_y - prev_avg_hip_y_1
                    
                    # Previous velocity (t-1 to t-2)
                    shoulder_vel_prev = prev_avg_shoulder_y_1 - prev_avg_shoulder_y_2
                    hip_vel_prev = prev_avg_hip_y_1 - prev_avg_hip_y_2
                    
                    # Calculate ACCELERATION (change in velocity)
                    shoulder_acceleration = shoulder_vel_current - shoulder_vel_prev
                    hip_acceleration = hip_vel_current - hip_vel_prev
                    avg_acceleration = (shoulder_acceleration + hip_acceleration) / 2
                    
                    # Calculate average body center velocity
                    body_center_y = (avg_shoulder_y + avg_hip_y) / 2
                    prev_body_center_y = (prev_avg_shoulder_y_1 + prev_avg_hip_y_1) / 2
                    body_velocity = body_center_y - prev_body_center_y
                    
                    # SLIP INDICATOR: Track wrist movement (hands up while body goes down)
                    # Get wrist positions if available
                    hands_raising = False
                    wrist_velocity = 0.0
                    if left_wrist[2] > 0.5 or right_wrist[2] > 0.5:
                        prev_left_wrist_1 = prev_kp_1[KEYPOINT_DICT['left_wrist']]
                        prev_right_wrist_1 = prev_kp_1[KEYPOINT_DICT['right_wrist']]
                        
                        wrist_velocities = []
                        # Check left wrist
                        if left_wrist[2] > 0.5 and prev_left_wrist_1[2] > 0.5:
                            left_wrist_vel = left_wrist[1] - prev_left_wrist_1[1]
                            wrist_velocities.append(left_wrist_vel)
                        
                        # Check right wrist
                        if right_wrist[2] > 0.5 and prev_right_wrist_1[2] > 0.5:
                            right_wrist_vel = right_wrist[1] - prev_right_wrist_1[1]
                            wrist_velocities.append(right_wrist_vel)
                        
                        if wrist_velocities:
                            wrist_velocity = np.mean(wrist_velocities)
                            # Hands moving UP (negative Y) while body moves DOWN (positive Y)
                            hands_raising = (wrist_velocity < -5) and (body_velocity > 10)
                    
                    # Indicator 1: Sudden acceleration towards floor
                    sudden_acceleration = avg_acceleration > 8.0  # Sudden change in velocity
                    
                    # Indicator 2: High downward velocity
                    rapid_descent = body_velocity > self.fall_params['y_velocity_threshold']
                    
                    # Indicator 3: Multiple keypoints moving downward together
                    all_moving_down = (shoulder_vel_current > 10 and hip_vel_current > 10)
                    
                    # Indicator 4: Torso angle change (loss of vertical posture)
                    current_torso_length = abs(avg_hip_y - avg_shoulder_y)
                    prev_torso_length = abs(prev_avg_hip_y_1 - prev_avg_shoulder_y_1)
                    torso_collapse = (prev_torso_length - current_torso_length) / prev_torso_length if prev_torso_length > 0 else 0
                    
                    # Indicator 5: Hip approaching floor level
                    frame_height = bbox[3] - bbox[1]
                    hip_from_bottom = bbox[3] - avg_hip_y
                    near_floor = hip_from_bottom < (frame_height * 0.4)
                    
                    # Indicator 6: SLIP SPECIFIC - Hands raising while body falls (key characteristic)
                    slip_pattern = hands_raising and rapid_descent
                    
                    # Detection Logic: Must have sudden change + downward movement
                    # Primary: Sudden acceleration with rapid descent
                    # Secondary: Multiple keypoints moving down rapidly with torso collapse  
                    # Tertiary: SLIP pattern - hands up while body down (most specific)
                    primary_fall = sudden_acceleration and rapid_descent
                    secondary_fall = all_moving_down and torso_collapse > 0.15
                    
                    if ((primary_fall or secondary_fall or slip_pattern) and near_floor):
                        # Boost confidence if slip pattern detected
                        confidence = 0.95 if slip_pattern else 0.93
                        
                        return {
                            'type': 'Sudden Falling',
                            'confidence': confidence,
                            'frame': self.frame_count,
                            'person_id': tracker_id,
                            'details': {
                                'acceleration': float(avg_acceleration),
                                'body_velocity': float(body_velocity),
                                'shoulder_velocity': float(shoulder_vel_current),
                                'hip_velocity': float(hip_vel_current),
                                'wrist_velocity': float(wrist_velocity),
                                'hands_raising': hands_raising,
                                'slip_pattern_detected': slip_pattern,
                                'torso_collapse_ratio': float(torso_collapse),
                                'near_floor': near_floor,
                                'sudden_acceleration': sudden_acceleration,
                                'method': 'acceleration_based'
                            }
                        }
        
        return None
    
    def _detect_fainting(self, persons: List[Dict]) -> Optional[Dict]:
        """
        Detect fainting: sudden fall followed by prolonged immobility
        """
        for tracker_id, tracker in self.trackers.items():
            # Check if person has fallen recently
            if tracker_id not in self.fainting_candidates:
                # Check for fall
                if len(tracker.bbox_history) >= 3:
                    prev_center_y = (tracker.bbox_history[-2][1] + tracker.bbox_history[-2][3]) / 2
                    curr_center_y = (tracker.bbox_history[-1][1] + tracker.bbox_history[-1][3]) / 2
                    y_velocity = curr_center_y - prev_center_y
                    
                    if y_velocity > self.fainting_params['initial_fall_threshold']:
                        # Fall detected, add to candidates
                        self.fainting_candidates[tracker_id] = (self.frame_count, 0)
            else:
                # Check for immobility after fall
                fall_frame, immobility_count = self.fainting_candidates[tracker_id]
                
                # Check velocity
                if len(tracker.velocity_history) > 0:
                    velocity = tracker.velocity_history[-1]
                    
                    if velocity < self.fainting_params['velocity_threshold']:
                        immobility_count += 1
                        self.fainting_candidates[tracker_id] = (fall_frame, immobility_count)
                        
                        # Check if immobile long enough
                        if immobility_count >= self.fainting_params['immobility_duration']:
                            del self.fainting_candidates[tracker_id]
                            return {
                                'type': 'Fainting/Collapse',
                                'confidence': 0.88,
                                'frame': self.frame_count,
                                'person_id': tracker_id,
                                'details': {
                                    'fall_frame': fall_frame,
                                    'immobility_duration': immobility_count,
                                    'velocity': float(velocity)
                                }
                            }
                    else:
                        # Person moved, remove from candidates
                        del self.fainting_candidates[tracker_id]
        
        return None
    
    def _detect_cardiac_distress(self, persons: List[Dict], frame: np.ndarray) -> Optional[Dict]:
        """
        Detect cardiac distress: hand to chest followed by fall
        """
        for person in persons:
            kp = person['keypoints']
            
            # Get relevant keypoints
            left_wrist = kp[KEYPOINT_DICT['left_wrist']]
            right_wrist = kp[KEYPOINT_DICT['right_wrist']]
            left_shoulder = kp[KEYPOINT_DICT['left_shoulder']]
            right_shoulder = kp[KEYPOINT_DICT['right_shoulder']]
            
            # Calculate chest center (between shoulders)
            if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                chest_center = (
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2
                )
                
                # Check if either hand is near chest
                hand_to_chest = False
                if left_wrist[2] > 0.5:
                    dist = np.sqrt(
                        (left_wrist[0] - chest_center[0])**2 + 
                        (left_wrist[1] - chest_center[1])**2
                    )
                    if dist < self.cardiac_params['hand_chest_distance']:
                        hand_to_chest = True
                
                if right_wrist[2] > 0.5:
                    dist = np.sqrt(
                        (right_wrist[0] - chest_center[0])**2 + 
                        (right_wrist[1] - chest_center[1])**2
                    )
                    if dist < self.cardiac_params['hand_chest_distance']:
                        hand_to_chest = True
                
                # Find corresponding tracker
                person_center = (
                    (person['bbox'][0] + person['bbox'][2]) / 2,
                    (person['bbox'][1] + person['bbox'][3]) / 2
                )
                
                for tracker_id, tracker in self.trackers.items():
                    tracker_center = tracker.get_center()
                    distance = np.sqrt(
                        (person_center[0] - tracker_center[0])**2 + 
                        (person_center[1] - tracker_center[1])**2
                    )
                    
                    if distance < 50:
                        if hand_to_chest:
                            # Track hand-to-chest gesture
                            if tracker_id not in self.cardiac_candidates:
                                self.cardiac_candidates[tracker_id] = self.frame_count
                        elif tracker_id in self.cardiac_candidates:
                            # Check if fall occurred after hand-to-chest
                            gesture_frame = self.cardiac_candidates[tracker_id]
                            frames_since_gesture = self.frame_count - gesture_frame
                            
                            if frames_since_gesture <= self.cardiac_params['fall_follows_within']:
                                # Check for fall
                                if len(tracker.bbox_history) >= 2:
                                    prev_center_y = (tracker.bbox_history[-2][1] + tracker.bbox_history[-2][3]) / 2
                                    curr_center_y = (tracker.bbox_history[-1][1] + tracker.bbox_history[-1][3]) / 2
                                    y_velocity = curr_center_y - prev_center_y
                                    
                                    if y_velocity > 15:
                                        del self.cardiac_candidates[tracker_id]
                                        return {
                                            'type': 'Cardiac Distress',
                                            'confidence': 0.82,
                                            'frame': self.frame_count,
                                            'person_id': tracker_id,
                                            'details': {
                                                'gesture_frame': gesture_frame,
                                                'fall_frame': self.frame_count,
                                                'sequence_detected': True
                                            }
                                        }
                            else:
                                # Timeout, remove candidate
                                del self.cardiac_candidates[tracker_id]
        
        return None
    
    def _detect_fire(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect fire using color-based heuristic (placeholder)
        Note: This is limited as YOLO-Pose focuses on humans
        """
        if not self.fire_params['enabled']:
            return None
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for fire colors (red/orange/yellow)
        lower = np.array(self.fire_params['color_range_lower'])
        upper = np.array(self.fire_params['color_range_upper'])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Calculate fire area
        fire_pixels = np.sum(mask > 0)
        
        if fire_pixels > self.fire_params['area_threshold']:
            return {
                'type': 'Fire',
                'confidence': 0.70,
                'frame': self.frame_count,
                'details': {
                    'fire_pixels': int(fire_pixels),
                    'detection_method': 'color_heuristic'
                }
            }
        
        return None
    
    def get_detection_summary(self) -> Dict:
        """Get summary of all detections"""
        summary = {
            'total_detections': len(self.detections),
            'by_type': {},
            'frames_processed': self.frame_count
        }
        
        for detection in self.detections:
            det_type = detection['type']
            if det_type not in summary['by_type']:
                summary['by_type'][det_type] = 0
            summary['by_type'][det_type] += 1
        
        return summary
