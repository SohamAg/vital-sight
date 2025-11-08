# Vital Sight - CCTV Distress Detection System

A real-time distress detection system that analyzes CCTV footage to identify people in distress (falls, abnormal poses, unusual behavior) and sends notifications.

## Features

- **Real-time Fall Detection**: Detects when a person falls using pose estimation
- **Abnormal Pose Detection**: Identifies unusual body positions indicating distress
- **Motion Analysis**: Tracks movement patterns to detect irregularities
- **Alert System**: Sends notifications when distress is detected
- **Video Processing**: Supports multiple video formats and live CCTV streams

## Technology Stack

- **OpenCV**: Video processing and computer vision
- **MediaPipe**: Real-time pose estimation
- **YOLOv8**: Person detection (optional enhancement)
- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Usage

```bash
# Process a video file
python main.py --input sample_videos/cctv_footage.mp4

# Use webcam
python main.py --input 0

# Process CCTV stream
python main.py --input rtsp://camera_ip:port/stream
```

### With Custom Settings

```bash
python main.py --input video.mp4 --confidence 0.7 --notification email
```

## How It Works

1. **Person Detection**: Identifies people in the video frame
2. **Pose Estimation**: Tracks 33 body keypoints using MediaPipe
3. **Distress Analysis**: 
   - Fall detection based on torso angle and position
   - Abnormal pose detection (lying down for extended periods)
   - Sudden movement changes
4. **Alert Generation**: Sends notifications via email/SMS/desktop

## Detection Criteria

### Fall Detection
- Rapid vertical displacement of body keypoints
- Horizontal body orientation (lying down)
- Sudden change in center of mass

### Distress Signals
- Person lying motionless for > 30 seconds
- Abnormal pose angles (< 45° from horizontal)
- Erratic movement patterns
- No movement detected for extended periods

## Configuration

Edit `config.py` to customize:
- Detection thresholds
- Notification methods
- Video processing parameters
- Alert sensitivity

## Sample Data

Sample CCTV footage will be downloaded automatically on first run, or you can add your own videos to the `sample_videos/` directory.

## Requirements

- Python 3.8+
- Webcam or video file for testing
- Internet connection (for downloading sample videos and models)
