## VitalSight Core — On-device Distress Detection (CLI + UI)

This is a small, modular, CPU-first pipeline that detects human distress using:

- YOLO (person boxes) → MediaPipe Pose (features) → Temporal FSM (events)
- Live overlays in a window and one-line JSON events to stdout
- Optional MP4 recording of the annotated frames
- Simple Tkinter UI to test webcam or any video file

No alerts, websockets, VLM, or servers in this core—by design.

### Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cp src/configs/example.env .env
```

### Run — CLI

```bash
# Webcam
python -m src.cli.main --source webcam

# Video file
python -m src.cli.main --source file --path ./data/samples/test.mp4

# Headless + save annotated MP4
python -m src.cli.main --source file --path ./data/samples/test.mp4 --no-display --save-debug ./runs/annotated.mp4
```

### Run — Simple UI

```bash
python -m src.ui.app
```
Use the UI to select webcam or choose a video file, start/stop the run, and optionally save an annotated MP4. Events appear in the log area.

### Configuration (.env)

See `src/configs/example.env` for defaults (input sizes, YOLO model/conf, thresholds):

- `ANGLE_THR_DEG` (supine if torso angle ≥ this, 0°=vertical, 90°=horizontal)
- `DROP_THR_PX` (sudden centroid Δy within 0.5s)
- `MOTION_THR` (mean landmark Δ below this => idle)
- `IMMOBILE_T_SEC` (consecutive idle seconds => immobile)
- `ALERT_SCORE_THR` (event emission threshold)

### Event JSON (stdout)

Each event is a compact one-line JSON, example:

```json
{"cam_id":"cam-0","track_id":"p-0","t0":12.3,"t1":13.4,"type":"collapse","score":0.83,"cues":["sudden_drop","supine_posture","immobile_8.2s"]}
```

### FAQ

- Why YOLO + Pose + Temporal?  
  YOLO robustly finds people. Pose extracts physical cues like torso orientation and motion. A short-memory temporal model reduces false positives from momentary noise.

- Does this repo send alerts or run a server?  
  No. The core emits events locally. Alerting/WS/VLM can be layered later.

- GPU required?  
  No. It runs on CPU; if CUDA is available, Ultralytics may use it automatically.

### Layout

```
src/
  common/      # pydantic schemas, small utils
  ingest/      # webcam/file reader normalized to size & FPS
  detect/      # YOLOv8 person detector + simple ID association
  pose/        # MediaPipe Pose features (torso angle, motion energy)
  reasoner/    # temporal FSM rule engine
  viz/         # overlays (boxes, features, cue badges, HUD)
  cli/         # CLI main
  ui/          # Tkinter app
```

### Notes

- Coordinates are pixel-space in the working (resized) frame. Origin is top-left.
- Timestamps are monotonic seconds since process start.
- Clean shutdown: capture release, window destruction, and writer close are handled.
