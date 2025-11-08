"""
Configuration file for Vital Sight - Distress Detection System
"""

# Detection Parameters
FALL_DETECTION_THRESHOLD = 45  # Angle in degrees to detect fall (horizontal orientation)
DISTRESS_TIME_THRESHOLD = 30   # Seconds a person must be in distress pose before alert
CONFIDENCE_THRESHOLD = 0.5     # Minimum confidence for pose detection
MOVEMENT_THRESHOLD = 10        # Pixel movement threshold to detect motion

# Pose Estimation Settings
POSE_MODEL_COMPLEXITY = 1      # 0=Lite, 1=Full, 2=Heavy (more accurate but slower)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Video Processing
FRAME_SKIP = 2                 # Process every Nth frame (1 = every frame)
DISPLAY_FPS = True             # Show FPS on output
SAVE_ALERTS = True             # Save frames where distress is detected
OUTPUT_DIR = "alerts"          # Directory to save alert images

# Alert System
ENABLE_DESKTOP_NOTIFICATIONS = True
ENABLE_SOUND_ALERTS = True
ENABLE_EMAIL_ALERTS = False    # Set to True and configure email settings
ENABLE_SMS_ALERTS = False      # Set to True and configure SMS settings

# Email Configuration (if ENABLE_EMAIL_ALERTS = True)
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_SENDER = "your-email@gmail.com"
EMAIL_PASSWORD = "your-app-password"
EMAIL_RECIPIENTS = ["recipient@email.com"]

# SMS Configuration (if ENABLE_SMS_ALERTS = True)
SMS_API_KEY = "your-api-key"
SMS_RECIPIENTS = ["+1234567890"]

# Detection Zones (optional - leave empty to monitor entire frame)
# Format: [(x1, y1, x2, y2), ...] where coordinates are fractions of frame size (0.0 to 1.0)
DETECTION_ZONES = []  # Example: [(0.2, 0.2, 0.8, 0.8)] monitors center 60% of frame

# Person Tracking
MAX_DISAPPEARED_FRAMES = 30    # Frames before considering person as left the scene
TRACK_HISTORY_LENGTH = 50      # Number of frames to keep in tracking history

# Visualization
DRAW_POSE_LANDMARKS = True
DRAW_BOUNDING_BOXES = True
DRAW_DETECTION_ZONES = True
ALERT_COLOR = (0, 0, 255)      # BGR format - Red
NORMAL_COLOR = (0, 255, 0)     # BGR format - Green
ZONE_COLOR = (255, 255, 0)     # BGR format - Cyan

# Performance
USE_GPU = False                # Set to True if GPU available (requires CUDA)
MAX_PERSONS_TO_TRACK = 10      # Maximum number of people to track simultaneously

# Debug Mode
DEBUG_MODE = True              # Print debug information
SAVE_DEBUG_FRAMES = False      # Save frames for debugging
