"""
Configuration file for VitalSight Emergency Detection System
"""

# System Configuration
CONFIDENCE_THRESHOLD = 0.5
YOLO_MODEL = "yolov8n-pose.pt"  # YOLOv8 Pose model
INPUT_SIZE = (640, 640)

# VLM Configuration (PLACEHOLDER - DISABLED)
# The VLM component is included in the structure but intentionally bypassed
VLM_ENABLED = False  # CRITICAL: Keep this False to use only YOLO-Pose + custom algorithms
VLM_MODEL_PATH = None  # Placeholder for future VLM integration
VLM_CONFIDENCE_THRESHOLD = 0.7  # Not used when VLM_ENABLED=False

# Emergency Action Detection Parameters
DETECTION_PARAMS = {
    'violence': {
        'min_subjects': 2,
        'velocity_threshold': 15.0,  # pixels per frame
        'distance_threshold': 100,   # pixels between subjects
        'duration_frames': 10        # sustained aggression frames
    },
    'fall': {
        'y_velocity_threshold': 20.0,  # pixels per frame downward
        'height_drop_ratio': 0.33,     # 1/3 of person height
        'detection_frames': 2          # frames to confirm fall
    },
    'fainting': {
        'initial_fall_threshold': 20.0,
        'immobility_duration': 30,     # frames
        'velocity_threshold': 2.0,     # low velocity = immobile
        'prone_angle_threshold': 45    # degrees from vertical
    },
    'cardiac': {
        'hand_chest_distance': 80,     # pixels
        'duration_frames': 15,         # hand on chest duration
        'fall_follows_within': 45      # frames after hand-to-chest
    },
    'fire': {
        # Placeholder for color/shape-based heuristic
        'enabled': False,  # Requires additional CV techniques
        'color_range_lower': (0, 50, 50),   # HSV
        'color_range_upper': (30, 255, 255),
        'area_threshold': 500
    }
}

# MCP Server Communication Settings
MCP_SERVER_CONFIG = {
    'phone_call_url': 'http://localhost:8000/api/call',
    'email_url': 'http://localhost:8000/api/email',
    'timeout': 5,  # seconds
    'retry_attempts': 3
}

# Alert Routing Configuration
ALERT_ROUTING = {
    'Fire': {
        'action': 'phone_call',
        'target': '9297602752'
    },
    'Cardiac Distress': {
        'action': 'phone_call',
        'target': '9297602752'
    },
    'Violence/Assault': {
        'action': 'email',
        'target': 'swapnil.sh2000@gmail.com'
    },
    'Sudden Falling': {
        'action': 'email',
        'target': 'swapnil.sh2000@gmail.com'
    },
    'Fainting/Collapse': {
        'action': 'email',
        'target': 'swapnil.sh2000@gmail.com'
    }
}

# Video Processing Settings
VIDEO_CONFIG = {
    'fps': 30,
    'buffer_size': 100,  # frames
    'alert_cooldown': 30,  # seconds between same alert
    'save_alert_images': True,
    'alert_image_dir': 'alerts'
}

# Visualization Settings
VISUALIZATION = {
    'show_keypoints': True,
    'show_skeleton': True,
    'show_bounding_box': True,
    'overlay_transparency': 0.6,
    'text_color': (0, 255, 0),
    'alert_color': (0, 0, 255),
    'keypoint_color': (255, 0, 0),
    'skeleton_color': (0, 255, 255)
}

# YOLO-Pose Keypoint Indices (COCO format)
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# YOLO-Pose Skeleton Connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]
