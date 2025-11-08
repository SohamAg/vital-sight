# Vital Sight - Usage Guide

Complete guide for using the CCTV Distress Detection System.

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd vital-sight

# Run the setup script (recommended)
./setup.sh

# OR manually install:
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Test with webcam
python3 main.py --input 0

# Test with a video file
python3 main.py --input path/to/video.mp4

# Download and test with sample video
python3 main.py --download-sample
```

## Command Line Options

```bash
python3 main.py [OPTIONS]
```

### Options

- `--input, -i <source>`: Input video source
  - `0` for webcam
  - File path for video files (`.mp4`, `.avi`, etc.)
  - RTSP URL for IP cameras (e.g., `rtsp://192.168.1.100:554/stream`)
  
- `--batch, -b <files...>`: Process multiple video files
  ```bash
  python3 main.py --batch video1.mp4 video2.mp4 video3.mp4
  ```

- `--download-sample`: Download sample video for testing

- `--confidence, -c <value>`: Set detection confidence threshold (0.0-1.0)
  ```bash
  python3 main.py --input 0 --confidence 0.7
  ```

- `--help`: Show help message

## Configuration

Edit `config.py` to customize the system behavior:

### Detection Parameters

```python
# Angle threshold for fall detection (degrees from vertical)
FALL_DETECTION_THRESHOLD = 45

# Time person must be in distress before alerting (seconds)
DISTRESS_TIME_THRESHOLD = 30

# Minimum confidence for pose detection
CONFIDENCE_THRESHOLD = 0.5
```

### Alert Configuration

```python
# Enable/disable alert types
ENABLE_DESKTOP_NOTIFICATIONS = True
ENABLE_SOUND_ALERTS = True
ENABLE_EMAIL_ALERTS = False
ENABLE_SMS_ALERTS = False
```

### Email Alerts

To enable email notifications:

1. Set `ENABLE_EMAIL_ALERTS = True`
2. Configure email settings:

```python
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_SENDER = "your-email@gmail.com"
EMAIL_PASSWORD = "your-app-password"  # Use app-specific password
EMAIL_RECIPIENTS = ["recipient1@email.com", "recipient2@email.com"]
```

**Note for Gmail:**
- Enable 2-factor authentication
- Generate an app-specific password at https://myaccount.google.com/apppasswords
- Use the app password instead of your regular password

### Detection Zones

Monitor specific areas of the frame:

```python
# Format: [(x1, y1, x2, y2), ...] with values 0.0-1.0
DETECTION_ZONES = [
    (0.2, 0.2, 0.8, 0.8)  # Monitor center 60% of frame
]

# Leave empty to monitor entire frame
DETECTION_ZONES = []
```

### Visualization

```python
DRAW_POSE_LANDMARKS = True      # Show skeleton overlay
DRAW_BOUNDING_BOXES = True      # Show boxes around people
DRAW_DETECTION_ZONES = True     # Show monitored zones
DISPLAY_FPS = True              # Show FPS counter
```

## Understanding Detections

### Fall Detection

The system detects falls using multiple indicators:

1. **Body Angle**: Detects horizontal body orientation (>45° from vertical)
2. **Vertical Position**: Tracks if person is on the ground
3. **Rapid Movement**: Detects sudden vertical displacement
4. **Sustained Position**: Person remains on ground for threshold time

### Alert Types

- **FALL_IN_PROGRESS**: Active fall detected (90% confidence)
- **PERSON_DOWN**: Person lying on ground for >30 seconds (60-95% confidence)
- **POSSIBLE_DISTRESS**: Person on ground but below time threshold (40% confidence)
- **IMMOBILE**: Person hasn't moved for extended period (up to 80% confidence)

### Visual Indicators

- **Green Box**: Normal activity detected
- **Red Box**: Distress detected - alert triggered
- **Skeleton Overlay**: Shows detected body pose
- **Angle & Position**: Debug info (if enabled)

## Interactive Controls

While the application is running:

- `q`: Quit the application
- `p`: Pause/resume video processing

## Use Cases

### 1. Home Care Monitoring

Monitor elderly or at-risk individuals:

```bash
# Use webcam
python3 main.py --input 0

# Use IP camera
python3 main.py --input rtsp://192.168.1.100:554/stream
```

Configure for sensitive detection:
```python
FALL_DETECTION_THRESHOLD = 40
DISTRESS_TIME_THRESHOLD = 20
ENABLE_DESKTOP_NOTIFICATIONS = True
```

### 2. Workplace Safety

Monitor industrial or construction areas:

```bash
python3 main.py --input cctv_feed.mp4
```

Set detection zones for high-risk areas:
```python
DETECTION_ZONES = [(0.3, 0.4, 0.7, 0.9)]  # Monitor specific work area
```

### 3. Healthcare Facilities

Monitor patient rooms or hallways:

```python
DISTRESS_TIME_THRESHOLD = 15  # Quick response
SAVE_ALERTS = True            # Document incidents
ENABLE_EMAIL_ALERTS = True    # Alert staff
```

### 4. Video Analysis

Analyze recorded footage:

```bash
# Single video
python3 main.py --input recorded_footage.mp4

# Batch processing
python3 main.py --batch day1.mp4 day2.mp4 day3.mp4
```

## Output Files

### Alert Images

When distress is detected and `SAVE_ALERTS = True`:

- Location: `alerts/` directory
- Format: `alert_{person_id}_{timestamp}_{count}.jpg`
- Content: Frame with distress overlay and confidence score

### Console Logs

Real-time alerts are logged to console:

```
============================================================
⚠️  DISTRESS ALERT TRIGGERED
============================================================
Person ID: 0
Type: PERSON_DOWN
Confidence: 85.0%
Body Angle: 78.3°
Vertical Position: 0.87
Time: 2025-01-08 02:30:45
============================================================
```

## Troubleshooting

### Camera Not Opening

```
Error: Could not open video source: 0
```

**Solutions:**
- Check camera permissions
- Try different camera index: `--input 1`, `--input 2`
- Verify camera is not in use by another application

### Low FPS / Performance

**Solutions:**
1. Increase frame skip:
   ```python
   FRAME_SKIP = 3  # Process every 3rd frame
   ```

2. Use lighter model:
   ```python
   POSE_MODEL_COMPLEXITY = 0  # Faster, less accurate
   ```

3. Reduce video resolution (resize input frames)

### False Positives

Too many false alerts?

**Solutions:**
1. Increase confidence threshold:
   ```python
   CONFIDENCE_THRESHOLD = 0.7
   ```

2. Increase distress time:
   ```python
   DISTRESS_TIME_THRESHOLD = 45  # Require longer duration
   ```

3. Adjust fall angle:
   ```python
   FALL_DETECTION_THRESHOLD = 50  # More horizontal required
   ```

### No Detections

Person not being detected?

**Solutions:**
1. Ensure adequate lighting
2. Check if person is in detection zone
3. Lower confidence threshold:
   ```python
   MIN_DETECTION_CONFIDENCE = 0.3
   ```

## Advanced Features

### RTSP Streams

Connect to IP cameras:

```bash
# Generic RTSP
python3 main.py --input rtsp://username:password@192.168.1.100:554/stream

# Common formats:
# - Hikvision: rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
# - Dahua: rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
# - Axis: rtsp://root:password@192.168.1.100/axis-media/media.amp
```

### Multiple Detection Zones

Monitor different areas with different rules:

```python
DETECTION_ZONES = [
    (0.0, 0.5, 0.3, 1.0),  # Left corridor
    (0.7, 0.5, 1.0, 1.0),  # Right corridor
    (0.3, 0.0, 0.7, 0.4),  # Central area
]
```

### Custom Alert Actions

Modify `notification_system.py` to add custom alert handlers:

```python
def send_custom_alert(self, distress_info, person_id):
    # Your custom alert logic
    # E.g., trigger IoT devices, call APIs, etc.
    pass
```

## Performance Optimization

### For Real-time Processing

```python
FRAME_SKIP = 1                 # Process all frames
POSE_MODEL_COMPLEXITY = 1      # Balanced accuracy/speed
USE_GPU = True                 # If available
```

### For Batch Processing

```python
FRAME_SKIP = 3                 # Process fewer frames
SAVE_ALERTS = True             # Save detections
DISPLAY_FPS = False            # Reduce overhead
```

## Best Practices

1. **Test Configuration**: Always test with known scenarios before deployment
2. **Adjust Thresholds**: Fine-tune based on your specific environment
3. **Monitor Alerts**: Review saved alert images to improve accuracy
4. **Regular Maintenance**: Update models and dependencies periodically
5. **Privacy Compliance**: Ensure compliance with local surveillance laws
6. **Backup Footage**: Keep original footage for verification

## Support

For issues or questions:
- Check configuration in `config.py`
- Enable debug mode: `DEBUG_MODE = True`
- Review alert images in `alerts/` directory
- Check console output for error messages

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Webcam or video file for input
- Optional: GPU for faster processing
- Optional: Internet for email/SMS alerts
