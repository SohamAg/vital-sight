from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import cv2
from dotenv import load_dotenv

from src.common.schemas import Detection, EventCandidate, PoseFeat
from src.common.utils import format_event_json, make_video_writer, now_s
from src.detect.yolo_person import YoloPersonDetector
from src.ingest.source import VideoSource
from src.pose.mediapipe_pose import PoseEstimator
from src.reasoner.fsm import TemporalReasoner
from src.viz.overlay import draw_overlays


def _env_or(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("VitalSight Core - CLI")
    p.add_argument("--source", choices=["webcam", "file"], default=_env_or("DEFAULT_SOURCE", "webcam"))
    p.add_argument("--path", type=str, default=_env_or("DEFAULT_VIDEO_PATH", "./data/samples/test.mp4"))
    p.add_argument("--width", type=int, default=int(_env_or("TARGET_WIDTH", "1280")))
    p.add_argument("--height", type=int, default=int(_env_or("TARGET_HEIGHT", "720")))
    p.add_argument("--fps", type=int, default=int(_env_or("TARGET_FPS", "15")))
    p.add_argument("--yolo-model", type=str, default=_env_or("YOLO_MODEL", "yolov8n.pt"))
    p.add_argument("--yolo-conf", type=float, default=float(_env_or("YOLO_CONF", "0.4")))
    p.add_argument("--angle-thr", type=float, default=float(_env_or("ANGLE_THR_DEG", "70")))
    p.add_argument("--drop-thr", type=float, default=float(_env_or("DROP_THR_PX", "60")))
    p.add_argument("--motion-thr", type=float, default=float(_env_or("MOTION_THR", "2.5")))
    p.add_argument("--immobile-sec", type=float, default=float(_env_or("IMMOBILE_T_SEC", "8.0")))
    p.add_argument("--score-thr", type=float, default=float(_env_or("ALERT_SCORE_THR", "0.8")))
    p.add_argument("--save-debug", type=str, default="")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main():
    # Load environment defaults if present
    load_dotenv(dotenv_path=os.getenv("VITALSIGHT_DOTENV", ".env"), override=False)
    args = parse_args()

    cap = None
    writer = None
    window_name = "VitalSight - Core"

    try:
        cap = VideoSource(
            kind=args.source,
            path=args.path if args.source == "file" else None,
            target_w=args.width,
            target_h=args.height,
            target_fps=args.fps,
        )
        detector = YoloPersonDetector(model_name=args.yolo_model, conf=args.yolo_conf)
        poser = PoseEstimator()
        reasoner = TemporalReasoner(
            drop_thr_px=args.drop_thr,
            angle_thr_deg=args.angle_thr,
            motion_thr=args.motion_thr,
            immobile_t=args.immobile_sec,
            alert_score_thr=args.score_thr,
            cam_id="cam-0",
        )

        if args.save_debug:
            writer = make_video_writer(args.save_debug, args.width, args.height, args.fps)
            if writer is None:
                print(f"Could not open writer for {args.save_debug}", file=sys.stderr)

        ema_fps = 0.0
        last_time = now_s()
        frame_idx = 0

        if not args.no_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            ok, frame, ts = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Inference
            detections: List[Detection] = detector.infer(frame)
            poses: Dict[str, PoseFeat] = {}
            for det in detections:
                pf = poser.infer_one(frame, det)
                if pf is not None:
                    poses[det.track_id] = pf

            events: List[EventCandidate] = reasoner.tick(ts=ts, detections=detections, poses=poses)
            # Output JSON events
            for ev in events:
                print(format_event_json(ev), flush=True)

            # FPS
            now = now_s()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            ema_fps = 0.9 * ema_fps + 0.1 * inst_fps if ema_fps > 0 else inst_fps
            last_time = now

            # Overlay and display
            draw_overlays(frame, frame_idx, ema_fps, detections, poses, events)
            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        # loop ended
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()


