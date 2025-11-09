# Installation Guide - VitalSight Emergency Detection System

## Prerequisites

Before installing VitalSight, ensure you have the following:

- **Python 3.8 or higher** (recommended: Python 3.9+)
- **pip** (Python package installer)
- **Webcam** (for real-time detection) or video files
- **4GB RAM minimum** (8GB+ recommended)
- **GPU** (optional but recommended for better performance)

## Step-by-Step Installation

### 1. Check Python Installation

First, verify Python is installed:

```bash
python3 --version
```

You should see output like `Python 3.9.x` or higher.

If Python is not installed:
- **macOS**: `brew install python3`
- **Ubuntu/Debian**: `sudo apt-get install python3 python3-pip`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

### 2. Clone or Download the Repository

```bash
# If using git
git clone <repository-url>
cd vital-sight

# Or download and extract the ZIP file
cd vital-sight
```

### 3. Create Virtual Environment (Recommended)

Creating a virtual environment isolates the project dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

Your terminal prompt should now show `(venv)` prefix.

### 4. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- OpenCV (computer vision)
- PyTorch (deep learning framework)
- Ultralytics (YOLO implementation)
- NumPy (numerical computing)
- And other dependencies

**Note**: Installation may take 5-10 minutes depending on your internet speed.

### 5. Verify Installation

Check if all packages are installed correctly:

```bash
python3 -c "import cv2, torch, ultralytics; print('All packages installed successfully!')"
```

### 6. Download YOLO-Pose Model

The YOLO-Pose model will be automatically downloaded on first run. You can trigger the download:

```bash
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n-pose.pt')"
```

This downloads the lightweight YOLO-Pose model (~6MB).

## Platform-Specific Notes

### macOS

**Webcam Permissions:**
When running the first time, macOS may ask for camera permissions. Grant permission in:
- System Preferences → Security & Privacy → Camera

**ARM (M1/M2/M3) Macs:**
PyTorch installation works seamlessly on Apple Silicon. GPU acceleration via Metal is supported.

### Linux

**Webcam Access:**
Ensure your user has permission to access the camera:

```bash
sudo usermod -a -G video $USER
```

Log out and log back in for changes to take effect.

**Display Server:**
For GUI display, ensure you have X11 or Wayland running.

### Windows

**Microsoft Visual C++ Redistributable:**
Some packages may require Visual C++ runtime. Download from:
https://support.microsoft.com/en-us/help/2977003/

**Webcam Drivers:**
Ensure webcam drivers are up to date via Device Manager.

## GPU Acceleration (Optional)

### NVIDIA GPU (CUDA)

For CUDA support with NVIDIA GPUs:

```bash
# Uninstall CPU-only PyTorch
pip uninstall torch torchvision

# Install CUDA-enabled PyTorch (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Check available CUDA versions at [PyTorch website](https://pytorch.org/get-started/locally/).

### AMD GPU (ROCm)

For AMD GPUs with ROCm support, follow AMD's ROCm installation guide.

### Apple Silicon (Metal)

PyTorch automatically uses Metal acceleration on M1/M2/M3 Macs.

## Verification

### Test the Installation

```bash
# List available sample videos
python3 main.py --list-samples

# Process a sample video
python3 main.py --sample 3
```

Press 'q' to quit when the video window appears.

### Expected Output

You should see:
1. Application banner
2. System information
3. Available sample videos list
4. Video processing window with pose detection
5. VLM status showing "DISABLED"

## Troubleshooting Installation Issues

### Issue: pip command not found

**Solution:**
```bash
# Try pip3 instead
pip3 install -r requirements.txt
```

### Issue: Permission denied errors

**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt
```

### Issue: SSL/Certificate errors during download

**Solution:**
```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools
```

### Issue: Conflicting package versions

**Solution:**
```bash
# Create fresh virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: OpenCV display issues

**Solution:**

On **Linux**:
```bash
sudo apt-get install python3-opencv libopencv-dev
```

On **macOS**:
```bash
brew install opencv
```

### Issue: Import errors for torch

**Solution:**
```bash
# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision
```

## Uninstallation

To completely remove VitalSight:

```bash
# Deactivate virtual environment
deactivate

# Remove project directory
cd ..
rm -rf vital-sight
```

## Next Steps

After successful installation:

1. **Read the README**: `README.md` for complete documentation
2. **Test with webcam**: `python3 main.py`
3. **Test with videos**: `python3 main.py --sample 1`
4. **Customize settings**: Edit `config.py` for your needs
5. **Check alerts**: Review saved alerts in `alerts/` directory

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.9+ |
| RAM | 4GB | 8GB+ |
| Storage | 2GB | 5GB+ |
| GPU | None (CPU) | NVIDIA/AMD |
| Webcam | 480p | 720p+ |

## Getting Help

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review `README.md` for detailed documentation
3. Ensure all dependencies are installed correctly
4. Check Python and package versions match requirements

## Updating

To update to the latest version:

```bash
# Pull latest changes (if using git)
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt
```

---

**Installation Time Estimate:** 10-15 minutes (including downloads)

**Note:** First run will download the YOLO-Pose model (~6MB) automatically.
