# VitalSight Emergency Detection System

A real-time emergency action detection system using YOLO-Pose for human pose estimation and custom rule-based algorithms to detect five specific emergency actions.

## 🎯 Overview

VitalSight is an intelligent video processing system that monitors for emergency situations in real-time. It leverages YOLO-Pose for accurate human pose estimation combined with custom detection algorithms to identify critical events and trigger appropriate communication actions via an MCP Server interface.

### Key Features

- **Real-time Detection**: Processes both webcam streams and video files
- **YOLO-Pose Integration**: Accurate human pose estimation with keypoint tracking
- **Custom Detection Algorithms**: Rule-based emergency action detection
- **MCP Communication**: Automated alert routing (phone calls and emails)
- **Visual Feedback**: Live visualization with bounding boxes, keypoints, and alerts
- **VLM Placeholder**: Intentionally disabled VLM component

## 🚨 Detectable Emergency Actions

| Emergency Action | Detection Method | Alert Type | Target |
|-----------------|------------------|------------|--------|
| **Violence/Assault** | Multi-person rapid movement + proximity | Email | swapnil.sh2000@gmail.com |
| **Sudden Falling** | Rapid downward movement detection | Email | swapnil.sh2000@gmail.com |
| **Fainting/Collapse** | Fall followed by prolonged immobility | Email | swapnil.sh2000@gmail.com |
| **Cardiac Distress** | Hand-to-chest gesture + subsequent fall | Phone Call | 9297602752 |
| **Fire** | Color-based heuristic (limited) | Phone Call | 9297602752 |

## 📋 Requirements

- Python 3.8 or higher
- Webcam (for real-time detection) or video files
- Sufficient GPU memory recommended for optimal performance

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vital-sight
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The YOLO-Pose model will be automatically downloaded on first run.

## 🚀 Usage

### Basic Usage

#### Webcam Detection (Default)
```bash
python main.py
```

#### Process Video File
```bash
python main.py --source sample_videos/fall.mp4
```

#### List Available Sample Videos
```bash
python main.py --list-samples
```

#### Process Sample Video by Index
```bash
python main.py --sample 1
```

### Command Line Options

```
usage: main.py [-h] [--source SOURCE] [--list-samples] [--sample SAMPLE]

VitalSight Emergency Detection System

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE, -s SOURCE
                        Video source (webcam ID or video file path). Default: webcam (0)
  --list-samples, -l    List available sample videos and exit
  --sample SAMPLE, -p SAMPLE
                        Process sample video by index (use --list-samples to see indices)
```

### During Execution

- Press **'q'** in the video window to quit
- Alert images are automatically saved to the `alerts/` directory
- Console displays real-time detection information and MCP alerts

## 📁 Project Structure

```
vital-sight/
├── main.py                    # Main entry point
├── video_processor.py         # Video processing pipeline
├── emergency_detector.py      # Emergency detection algorithms
├── mcp_interface.py           # MCP Server communication
├── vlm_placeholder.py         # VLM placeholder (DISABLED)
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
├── sample_videos/             # Sample video files
│   ├── fall.mp4
│   ├── LIVELEAK_LiveLeak3_struggle_1.mp4
│   ├── SPHAR_p17faint_lying_down_1.mp4
│   ├── UCFCRIME_Shooting004_gun_1.mp4
│   └── YOUTUBE_YouTubeCCTV160_struggle_2.mp4
└── alerts/                    # Alert images (auto-generated)
```

## 🔍 Detection Algorithms

### 1. Violence/Assault Detection

**Heuristics:**
- Detects 2+ people in close proximity (< 100 pixels)
- Rapid, aggressive keypoint velocity (> 15 pixels/frame)
- Sustained aggressive behavior over multiple frames

### 2. Sudden Falling Detection

**Heuristics:**
- Rapid downward movement of person's center (> 20 pixels/frame)
- Drop exceeds 1/3 of person's height
- Confirmed within 2 frames

### 3. Fainting/Collapse Detection

**Heuristics:**
- Initial fall detection (similar to sudden falling)
- Followed by prolonged immobility (30+ frames)
- Low keypoint velocity (< 2 pixels/frame) in prone position

### 4. Cardiac Distress Detection

**Heuristics:**
- Hand-to-chest gesture detection (wrist within 80 pixels of chest)
- Sustained hand position (15+ frames)
- Followed by fall within 45 frames

### 5. Fire Detection (Limited)

**Heuristics:**
- Color-based HSV detection (red/orange/yellow)
- Area threshold (> 500 pixels)
- Note: Currently disabled by default; requires additional CV techniques

## ⚙️ Configuration

All system parameters can be adjusted in `config.py`:

### Detection Parameters

```python
DETECTION_PARAMS = {
    'violence': {
        'min_subjects': 2,
        'velocity_threshold': 15.0,
        'distance_threshold': 100,
        'duration_frames': 10
    },
    'fall': {
        'y_velocity_threshold': 20.0,
        'height_drop_ratio': 0.33,
        'detection_frames': 2
    },
    # ... etc
}
```

### Alert Routing

```python
ALERT_ROUTING = {
    'Fire': {
        'action': 'phone_call',
        'target': '9297602752'
    },
    'Cardiac Distress': {
        'action': 'phone_call',
        'target': '9297602752'
    },
    # ... etc
}
```

### VLM Configuration (DISABLED)

```python
VLM_ENABLED = False  # CRITICAL: Keep False to use only YOLO-Pose + custom algorithms
```

## 🎨 Visualization

The system provides real-time visual feedback with:

- **Green bounding boxes** around detected persons
- **Red/Blue keypoints** showing body joints
- **Cyan skeleton connections** between keypoints
- **Red alert text** for detected emergencies
- **Frame information** (count, FPS)
- **VLM status** indicator (showing disabled state)

## 🔔 MCP Server Integration

The system includes a mock MCP Server interface for demonstration:

### Phone Call Alerts
```python
# Triggered for: Fire, Cardiac Distress
# Target: 9297602752
```

### Email Alerts
```python
# Triggered for: Violence/Assault, Sudden Falling, Fainting/Collapse
# Target: swapnil.sh2000@gmail.com
```

### Alert Cooldown

- 30-second cooldown between same alert types
- Prevents alert spam
- Configurable in `config.py`

## 📊 Output

### Console Output

```
[ALERT] Sudden Falling detected at frame 145
[ALERT] Confidence: 0.90
[ALERT] Details: {'y_velocity': 25.3, 'height': 180.5, 'drop_detected': True}

============================================================
[MCP] EMAIL ALERT
============================================================
Emergency Type: Sudden Falling
Target Email: swapnil.sh2000@gmail.com
Timestamp: 2025-01-08 21:30:15
Frame ID: 145
[MCP] Status: EMAIL SENT
============================================================
```

### Alert Images

Automatically saved to `alerts/` directory with format:
```
alert_0_20250108_213015_145.jpg
```

### Processing Summary

```
============================================================
PROCESSING SUMMARY
============================================================
Total frames processed: 1250
Total detections: 3

Detections by type:
  - Sudden Falling: 2
  - Violence/Assault: 1

Total MCP alerts sent: 3
Alert images saved: 3
============================================================
```

## 🧪 Testing with Sample Videos

The system includes sample videos for testing different scenarios:

1. **fall.mp4** - Falling detection
2. **LIVELEAK_LiveLeak3_struggle_1.mp4** - Violence/assault scenarios
3. **SPHAR_p17faint_lying_down_1.mp4** - Fainting/collapse scenarios
4. **UCFCRIME_Shooting004_gun_1.mp4** - Complex movement scenarios
5. **YOUTUBE_YouTubeCCTV160_struggle_2.mp4** - Multi-person scenarios

```bash
# List all sample videos
python main.py --list-samples

# Process a specific sample
python main.py --sample 1
```

## 🛠️ Troubleshooting

### Issue: Model Download Fails
**Solution:** Ensure stable internet connection. The YOLO-Pose model will auto-download on first run.

### Issue: Webcam Not Detected
**Solution:** 
- Check webcam permissions
- Try different webcam ID: `python main.py --source 1`

### Issue: Low FPS
**Solution:**
- Reduce input resolution in `config.py`
- Ensure GPU acceleration is available
- Close other resource-intensive applications

### Issue: False Detections
**Solution:** Adjust detection thresholds in `config.py`:
```python
CONFIDENCE_THRESHOLD = 0.6  # Increase for fewer false positives
```

## 🔐 Security & Privacy

- All video processing is performed locally
- No data is transmitted to external servers (MCP interface is mocked)
- Alert images contain only detected emergency frames
- Configurable data retention policies

## 🎯 Design Philosophy: VLM Placeholder Approach

This system intentionally includes a **VLM (Visual Language Model) placeholder** that is **disabled by default**. This design decision ensures:

1. **Simplicity**: Detection relies solely on YOLO-Pose + custom algorithms
2. **Performance**: No VLM overhead, faster processing
3. **Transparency**: Clear separation of detection methods
4. **Extensibility**: Easy to enable VLM in future if needed

The VLM placeholder is maintained in the codebase (`vlm_placeholder.py`) but is bypassed during all detection operations.

## 📝 License

This project is provided for educational and demonstration purposes.

## 👥 Authors

VitalSight Development Team

## 🙏 Acknowledgments

- **Ultralytics YOLOv8** for the pose estimation model
- **OpenCV** for video processing capabilities
- Sample videos from various public datasets

## 📧 Contact

For questions or issues, please contact: swapnil.sh2000@gmail.com

---

**Note:** This is a demonstration system. The MCP Server interface is mocked for testing purposes. In production, integrate with actual communication APIs.
