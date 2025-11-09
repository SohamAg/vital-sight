# Quick Start Guide - VitalSight

Get started with VitalSight Emergency Detection System in 5 minutes!

## ⚡ Fast Track Installation

```bash
# 1. Navigate to project directory
cd vital-sight

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Run the system
python3 main.py
```

That's it! Press 'q' to quit.

## 🎬 Quick Examples

### Example 1: Use Webcam (Real-time Detection)

```bash
python3 main.py
```

This opens your webcam and starts detecting emergency actions in real-time.

### Example 2: Process a Video File

```bash
python3 main.py --source sample_videos/fall.mp4
```

Process a pre-recorded video file for emergency detection.

### Example 3: List & Select Sample Videos

```bash
# List available sample videos
python3 main.py --list-samples

# Process sample video by index
python3 main.py --sample 1
```

## 📺 What You'll See

When running, you'll see:

1. **Video Window** with:
   - Green bounding boxes around people
   - Red/Blue keypoints on body joints
   - Cyan skeleton connections
   - Red alert text when emergencies detected

2. **Console Output** with:
   - Frame-by-frame processing info
   - Detection alerts
   - MCP notification logs

3. **Saved Alert Images** in `alerts/` directory

## 🚨 Emergency Actions Detected

| Action | What Triggers It | Alert Method |
|--------|-----------------|--------------|
| Violence/Assault | 2+ people, rapid movement, close proximity | Email |
| Sudden Falling | Person drops rapidly | Email |
| Fainting/Collapse | Fall + prolonged immobility | Email |
| Cardiac Distress | Hand-to-chest + fall | Phone Call |
| Fire | Red/orange color detection (limited) | Phone Call |

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |

## 📁 Output Files

After running, check:

```bash
# Alert images with detection info
alerts/alert_0_20250108_213015_145.jpg
alerts/alert_1_20250108_213045_289.jpg
...
```

## 🎛️ Quick Configuration

Edit `config.py` to adjust:

```python
# Detection sensitivity
CONFIDENCE_THRESHOLD = 0.5  # Higher = fewer false positives

# Alert cooldown (seconds)
VIDEO_CONFIG = {
    'alert_cooldown': 30,  # Time between same alert type
}

# VLM Status (Keep False!)
VLM_ENABLED = False  # Uses YOLO-Pose + custom algorithms only
```

## 🔧 Common Commands

```bash
# Use webcam
python3 main.py

# Process specific video
python3 main.py --source /path/to/video.mp4

# Process using webcam ID 1 (if you have multiple cameras)
python3 main.py --source 1

# List sample videos
python3 main.py --list-samples

# Process sample video 3
python3 main.py --sample 3

# Get help
python3 main.py --help
```

## 💡 Tips for Best Results

1. **Good Lighting**: Ensure adequate lighting for better pose detection
2. **Camera Position**: Position camera to capture full body view
3. **Clear View**: Avoid obstructions between camera and subjects
4. **Stable Camera**: Use stable mount to reduce motion blur
5. **Distance**: Keep subjects 2-10 meters from camera

## 🐛 Quick Troubleshooting

### No video window appears
**Fix**: Check webcam permissions, try different camera ID
```bash
python3 main.py --source 1
```

### Low FPS / Laggy
**Fix**: Close other applications, reduce video resolution
```python
# In config.py
INPUT_SIZE = (416, 416)  # Smaller = faster
```

### False detections
**Fix**: Increase confidence threshold
```python
# In config.py
CONFIDENCE_THRESHOLD = 0.6  # Was 0.5
```

### Webcam not found
**Fix**: Check device permissions and try:
```bash
# List video devices (Linux/Mac)
ls /dev/video*

# Test with different ID
python3 main.py --source 0
python3 main.py --source 1
```

## 📊 Understanding Console Output

```
[VideoProcessor] Loading YOLO-Pose model: yolov8n-pose.pt
[VLM] VLM component is DISABLED (as per design)
[Detector] Emergency detector initialized
[VideoProcessor] Initialized successfully

============================================================
VIDEO PROCESSING STARTED
============================================================
Resolution: 1280x720
FPS: 30
Press 'q' to quit
============================================================

[ALERT] Sudden Falling detected at frame 145
[ALERT] Confidence: 0.90
[ALERT] Details: {'y_velocity': 25.3, 'height': 180.5}

============================================================
[MCP] EMAIL ALERT
============================================================
Emergency Type: Sudden Falling
Target Email: swapnil.sh2000@gmail.com
Timestamp: 2025-01-08 21:30:15
[MCP] Status: EMAIL SENT
============================================================
```

## 🎓 Learning Path

1. ✅ **Quick Start** (You are here!)
2. 📖 **Full Documentation**: Read `README.md`
3. 🔧 **Customization**: Explore `config.py`
4. 🧪 **Sample Videos**: Test different scenarios
5. 💻 **Code Deep Dive**: Study detection algorithms

## ⚡ Performance Tips

### For Faster Processing
```python
# In config.py
YOLO_MODEL = "yolov8n-pose.pt"  # Nano (fastest)
INPUT_SIZE = (416, 416)  # Smaller resolution
```

### For Better Accuracy
```python
# In config.py
YOLO_MODEL = "yolov8m-pose.pt"  # Medium (more accurate)
CONFIDENCE_THRESHOLD = 0.6  # Higher threshold
```

## 📞 MCP Alert Targets

The system sends alerts to:

- **Phone Calls**: `9297602752` (Fire, Cardiac Distress)
- **Emails**: `swapnil.sh2000@gmail.com` (Other emergencies)

To change these, edit `ALERT_ROUTING` in `config.py`.

## 🔄 Next Steps

After your first run:

1. **Review Alerts**: Check `alerts/` directory for saved images
2. **Test Scenarios**: Try different sample videos
3. **Customize Settings**: Adjust detection parameters in `config.py`
4. **Read Full Docs**: See `README.md` for detailed information

## 📚 Additional Resources

- **Full Documentation**: `README.md`
- **Installation Guide**: `INSTALL.md`
- **Configuration**: `config.py`
- **Detection Code**: `emergency_detector.py`

## 🎯 System Architecture Summary

```
Video Input (Webcam/File)
    ↓
YOLO-Pose Detection (Keypoints)
    ↓
Emergency Detection Algorithms
    ↓
Alert Triggered?
    ↓ Yes
MCP Server Interface (Phone/Email)
    ↓
Save Alert Image & Log
```

## ❓ Need Help?

1. Check `README.md` for detailed documentation
2. Review `INSTALL.md` for installation issues
3. Ensure all dependencies are installed: `pip3 list`
4. Test with sample videos first: `python3 main.py --sample 1`

---

**Ready to detect emergencies?** Run `python3 main.py` and press 'q' to quit!

🎥 **Note**: VLM is intentionally DISABLED - detection uses YOLO-Pose + custom algorithms only.
