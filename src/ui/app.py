from __future__ import annotations

import threading
import queue
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Dict, List, Optional

import cv2
from PIL import Image, ImageTk
from dotenv import load_dotenv

from src.common.schemas import Detection, EventCandidate, PoseFeat
from src.common.utils import format_event_json, make_video_writer, now_s
from src.detect.yolo_person import YoloPersonDetector
from src.ingest.source import VideoSource
from src.pose.mediapipe_pose import PoseEstimator
from src.reasoner.fsm import TemporalReasoner
from src.viz.overlay import draw_overlays
import os


class VitalSightApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VitalSight - UI")
        self.geometry("1000x700")

        load_dotenv(dotenv_path=os.getenv("VITALSIGHT_DOTENV", ".env"), override=False)

        # Controls
        self.source_var = tk.StringVar(value=os.getenv("DEFAULT_SOURCE", "webcam"))
        self.path_var = tk.StringVar(value=os.getenv("DEFAULT_VIDEO_PATH", "./data/samples/test.mp4"))
        self.width_var = tk.IntVar(value=int(os.getenv("TARGET_WIDTH", "1280")))
        self.height_var = tk.IntVar(value=int(os.getenv("TARGET_HEIGHT", "720")))
        self.fps_var = tk.IntVar(value=int(os.getenv("TARGET_FPS", "15")))
        self.save_path_var = tk.StringVar(value="")

        # Detector / reasoner config
        self.yolo_model = os.getenv("YOLO_MODEL", "yolov8n.pt")
        self.yolo_conf = float(os.getenv("YOLO_CONF", "0.4"))
        self.angle_thr = float(os.getenv("ANGLE_THR_DEG", "70"))
        self.drop_thr = float(os.getenv("DROP_THR_PX", "60"))
        self.motion_thr = float(os.getenv("MOTION_THR", "2.5"))
        self.immobile_sec = float(os.getenv("IMMOBILE_T_SEC", "8"))
        self.score_thr = float(os.getenv("ALERT_SCORE_THR", "0.8"))

        self._build_layout()
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._frame_queue: "queue.Queue" = queue.Queue(maxsize=2)
        self._writer = None

    def _build_layout(self):
        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(controls, text="Source:").pack(side=tk.LEFT)
        ttk.Radiobutton(controls, text="Webcam", variable=self.source_var, value="webcam").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(controls, text="File", variable=self.source_var, value="file").pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Choose File", command=self._choose_file).pack(side=tk.LEFT, padx=4)
        ttk.Entry(controls, textvariable=self.path_var, width=40).pack(side=tk.LEFT, padx=4)

        ttk.Label(controls, text="W×H:").pack(side=tk.LEFT, padx=6)
        ttk.Entry(controls, textvariable=self.width_var, width=6).pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.height_var, width=6).pack(side=tk.LEFT)
        ttk.Label(controls, text="FPS:").pack(side=tk.LEFT, padx=6)
        ttk.Entry(controls, textvariable=self.fps_var, width=4).pack(side=tk.LEFT)

        ttk.Button(controls, text="Start", command=self.start).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Stop", command=self.stop).pack(side=tk.LEFT)

        ttk.Button(controls, text="Save MP4...", command=self._choose_save).pack(side=tk.LEFT, padx=8)
        ttk.Entry(controls, textvariable=self.save_path_var, width=30).pack(side=tk.LEFT)

        # Canvas for video
        self.canvas = tk.Label(self)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Events log
        self.events_txt = tk.Text(self, height=8)
        self.events_txt.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

    def _choose_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.webm *.avi"), ("All files", "*.*")])
        if path:
            self.path_var.set(path)
            self.source_var.set("file")

    def _choose_save(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if path:
            self.save_path_var.set(path)

    def start(self):
        if self._running:
            return
        self._running = True
        self.events_txt.delete("1.0", tk.END)
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()
        self.after(10, self._refresh_canvas)

    def stop(self):
        self._running = False

    def _refresh_canvas(self):
        try:
            frame = self._frame_queue.get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)
        if self._running:
            self.after(10, self._refresh_canvas)

    def _run_loop(self):
        source = self.source_var.get()
        path = self.path_var.get()
        w, h, fps = self.width_var.get(), self.height_var.get(), self.fps_var.get()
        writer = None
        if self.save_path_var.get():
            writer = make_video_writer(self.save_path_var.get(), w, h, fps)
        self._writer = writer
        try:
            cap = VideoSource(
                kind=source,
                path=path if source == "file" else None,
                target_w=w,
                target_h=h,
                target_fps=fps,
            )
            det = YoloPersonDetector(model_name=self.yolo_model, conf=self.yolo_conf)
            pose = PoseEstimator()
            fsm = TemporalReasoner(
                drop_thr_px=self.drop_thr,
                angle_thr_deg=self.angle_thr,
                motion_thr=self.motion_thr,
                immobile_t=self.immobile_sec,
                alert_score_thr=self.score_thr,
                cam_id="cam-0",
            )
            frame_idx = 0
            last_t = now_s()
            ema_fps = 0.0
            while self._running:
                ok, frame, ts = cap.read()
                if not ok:
                    break
                frame_idx += 1
                detections: List[Detection] = det.infer(frame)
                poses: Dict[str, PoseFeat] = {}
                for d in detections:
                    pf = pose.infer_one(frame, d)
                    if pf is not None:
                        poses[d.track_id] = pf
                events: List[EventCandidate] = fsm.tick(ts, detections, poses)
                for ev in events:
                    self.events_txt.insert(tk.END, format_event_json(ev) + "\n")
                    self.events_txt.see(tk.END)
                now = now_s()
                fps_inst = 1.0 / max(1e-6, now - last_t)
                ema_fps = 0.9 * ema_fps + 0.1 * fps_inst if ema_fps > 0 else fps_inst
                last_t = now
                draw_overlays(frame, frame_idx, ema_fps, detections, poses, events)
                if writer is not None:
                    writer.write(frame)
                # push to UI queue
                try:
                    if not self._frame_queue.full():
                        self._frame_queue.put_nowait(frame.copy())
                except Exception:
                    pass
        except Exception as e:
            self.events_txt.insert(tk.END, f"Error: {e}\n")
            self.events_txt.see(tk.END)
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass


def main():
    app = VitalSightApp()
    app.mainloop()


if __name__ == "__main__":
    main()


