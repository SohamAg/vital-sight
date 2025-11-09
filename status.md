# 🧠 VitalSight: AI-Driven Hybrid Intelligence System for Human Distress Detection

## 🎯 **Objective**
VitalSight is an **AI-powered, real-time surveillance system** designed to monitor **CCTV and webcam feeds** for signs of **human distress** and trigger **autonomous, context-aware responses**.  
It aims to augment traditional monitoring systems by detecting emergencies like falls, respiratory distress, immobility, violence, or fire without requiring human supervision.

---

## ⚙️ **System Architecture**

### **1. Input Layer**
- Accepts both **live webcam** and **pre-recorded CCTV feeds**.  
- Streams are processed frame-by-frame in real time.  
- Supports multiple concurrent camera feeds.

### **2. Detection Layer**
- Uses **Ultralytics YOLO 11** for high-speed **object/person detection** on GPU (CUDA 12 / FP16).  
- Optional **YOLO 11-pose** or **MediaPipe** backend for **pose estimation** to extract:
  - Body posture (standing, lying, crouched)
  - Hand-to-chest gestures
  - Torso angle and orientation

### **3. Classification Layer (Heuristic Engine)**
The system assigns probabilistic scores to five key event categories (multi-label, frame-wise):
| Category | Key Signal / Heuristic |
|-----------|------------------------|
| **Fire** | HSV color threshold for orange/yellow flame regions |
| **Fall** | Sudden downward motion in person’s center coordinates |
| **Immobility** | Sustained low motion variance over time |
| **Respiratory Distress** | Reduced hand–chest distance via pose |
| **Violence / Panic** | High motion + low proximity among multiple people |

All detections are person-gated (no human → no false positives) and stabilized with frame averaging and confidence debouncing.

---

## 🧩 **Pipeline Summary**
```
Video/Webcam Feed
        ↓
YOLO 11 (object/person detection)
        ↓
Pose Estimation (YOLO-pose / MediaPipe)
        ↓
Heuristic Analysis Engine
        ↓
Event Classification (multi-label)
        ↓
Visual Overlay + Event Stream
        ↓
Alert System (email / voice / reasoning)
```

---

## 🧰 **Technology Stack**
| Component | Tool / Framework |
|------------|------------------|
| **Detection** | Ultralytics YOLO 11 (PyTorch + CUDA) |
| **Pose Estimation** | YOLO 11-pose / MediaPipe |
| **Heuristics & Logic** | NumPy, OpenCV, Python |
| **Inference** | PyTorch 2.4 (CUDA 12, FP16) |
| **Alerts (Planned)** | Twilio, SendGrid, ElevenLabs |
| **Reasoning (Planned)** | Dedalus Labs API / Gemini VLM |
| **UI (Planned)** | Streamlit dashboard |

---

## 🧱 **Project Structure**
```
vitalsight/
├── config.yaml           # thresholds, runtime settings
├── requirements.txt      # dependencies (YOLO 11, CUDA, MediaPipe)
├── main.py               # entry point
└── edge/
    ├── detector_v2.py    # YOLO detection + heuristics
    └── pose.py           # pose extraction + feature computation
```

---

## 🚀 **Next Development Goals**
- Add **temporal logic** for sustained event confirmation.  
- Fuse **pose + motion features** for higher accuracy.  
- Introduce **alert orchestration** (email, voice call, chatbot).  
- Integrate **Dedalus reasoning agent** for contextual summaries.  
- Build a **Streamlit dashboard** for visualization and multi-camera management.

---

## 💻 **Performance (RTX 4060)**
- YOLO 11n: 40–120 FPS @ 640×640  
- YOLO 11-pose: 30–90 FPS (top 2–3 humans)  
- Full pipeline real-time on single RTX 4060 GPU  
- Supports multiple 720p streams simultaneously
