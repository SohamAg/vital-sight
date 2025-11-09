# edge/detector_v2.py
import cv2, time, math, yaml
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch
from .pose import PoseEstimator, keypose_feats

POSE_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (5, 11), (6, 12), (11, 12),  # torso
    (0, 1), (1, 2), (2, 3), (3, 4),  # head chain
    (0, 5), (0, 6)  # shoulders to nose
]

def now_ms(): return int(time.time()*1000)

class Debouncer:
    def __init__(self, enter=0.55, exit=0.35, min_frames=8):
        self.enter, self.exit = enter, exit
        self.min_frames = min_frames
        self.active = False
        self.count = 0
        self.score_smooth = 0.0
    def update(self, score):
        # simple EMA + hysteresis
        self.score_smooth = 0.7*self.score_smooth + 0.3*score
        if not self.active:
            self.count = self.count + 1 if self.score_smooth >= self.enter else 0
            if self.count >= self.min_frames:
                self.active = True
        else:
            if self.score_smooth < self.exit:
                self.active, self.count = False, 0
        return self.active, self.score_smooth

class VitalSightV2:
    def __init__(self, cfg_path="config.yaml"):
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        y = self.cfg["yolo"]
        device = self._resolve_device(self.cfg["runtime"]["device"])
        self.device = device

        self.model = YOLO(y["model"])
        self.model.to(device)

        # Optional: enable cuDNN autotune for speed
        if device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.conf = y["conf"]; self.iou = y["iou"]; self.classes = y.get("classes",[0])
        self.input_size = self.cfg["runtime"]["input_size"]
        self.profile = self.cfg.get("profile","webcam")

        # Fire detection via YOLO classes (if enabled)
        fire_cfg = self.cfg["logic"]["fire"]
        self.fire_mode = fire_cfg.get("detection_mode", "yolo")  # "hsv" or "yolo"
        self.fire_model = None
        self.fire_classes = []
        if self.fire_mode == "yolo":
            fire_weights = fire_cfg.get("yolo_model", "yolo11n.pt")
            self.fire_model = YOLO(fire_weights)
            self.fire_model.to(device)
            self.fire_classes = fire_cfg.get("yolo_classes", [])

        # debouncers per label
        hys = self.cfg["hysteresis"]
        self.db = {
            # allow fall to have its own enter/exit if provided
            "fall": Debouncer(hys.get("fall_enter", hys["enter"]),
                              hys.get("fall_exit", hys["exit"]),
                              self.cfg["logic"]["fall"]["debounce_f"]),
            "respiratory_distress": Debouncer(hys["enter"], hys["exit"], self.cfg["logic"]["resp"]["debounce_f"]),
            "violence_panic": Debouncer(hys["enter"], hys["exit"], self.cfg["logic"]["violence"]["debounce_f"]),
            "fire": Debouncer(hys["enter"], hys["exit"], self.cfg["logic"]["fire"]["persist_f"]),
            "severe_injury": Debouncer(hys["enter"], hys["exit"], self.cfg["logic"]["severe_injury"]["debounce_f"]),
        }

        # motion buffers
        self.track_history = deque(maxlen=30)  # list of centers for all persons
        self.person_boxes = []                 # last-frame person boxes (resized frame coords)
        self.prev_person_boxes = []
        self.none_active = True

        # pose
        p = self.cfg["pose"]
        self.pose_enabled = bool(p["enabled"])
        self.pose_backend = p["backend"]
        self.pose_max_people = p["max_people"]
        self.min_box_area_frac = p["min_box_area_frac"]
        self.pose_est = PoseEstimator(p["backend"]) if self.pose_enabled else None

        self.last_fps_t = time.time()
        self.fps = 0.0

    def _resolve_device(self, want):
        if want == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return want

    def _preprocess(self, frame):
        return cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

    def _detect_persons(self, frame):
        res = self.model.predict(
                frame,
                imgsz=self.input_size,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                half=(self.device == "cuda"),   # run inference in FP16 safely
                verbose=False
            )[0]
        xyxy = res.boxes.xyxy
        if xyxy is None or len(xyxy)==0:
            self.prev_person_boxes = self.person_boxes.copy()
            self.person_boxes = []
            return res, []
        boxes = xyxy.cpu().numpy().tolist()
        self.prev_person_boxes = self.person_boxes.copy()
        self.person_boxes = boxes
        return res, self.person_boxes

    def _motion_value(self):
        if len(self.track_history) < 2: return 0.0
        prev = self.track_history[-2]
        curr = self.track_history[-1]
        if not prev or not curr: return 0.0
        # nearest-neighbor mean displacement
        deltas = []
        for cx,cy in prev:
            d = min(math.hypot(cx-x2, cy-y2) for (x2,y2) in curr) if curr else 0
            deltas.append(d)
        return float(np.mean(deltas)) if deltas else 0.0

    def _centers(self, boxes):
        cs = []
        for x1,y1,x2,y2 in boxes:
            cs.append(((x1+x2)/2.0, (y1+y2)/2.0))
        return cs

    def _match_boxes(self, prev_boxes, curr_boxes):
        if not prev_boxes or not curr_boxes:
            return []
        max_dist = float(self.cfg["logic"]["fall"].get("match_dist_norm", 0.35))
        matches = []
        used_curr = set()
        for i, pb in enumerate(prev_boxes):
            pcx = (pb[0] + pb[2]) * 0.5
            pcy = (pb[1] + pb[3]) * 0.5
            best_j = -1
            best_dist = None
            for j, cb in enumerate(curr_boxes):
                if j in used_curr:
                    continue
                ccx = (cb[0] + cb[2]) * 0.5
                ccy = (cb[1] + cb[3]) * 0.5
                dist = math.hypot(pcx - ccx, pcy - ccy) / float(self.input_size)
                if dist > max_dist:
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j >= 0:
                used_curr.add(best_j)
                matches.append((i, best_j))
        return matches

    def _box_orientation_score(self, boxes):
        if not boxes:
            return 0.0
        thr = self.cfg["logic"]["fall"].get("width_height_ratio", 1.6)
        baseline = 1.0
        norm = max(1e-6, thr - baseline)
        scores = []
        for x1, y1, x2, y2 in boxes:
            w = max(1.0, abs(x2 - x1))
            h = max(1.0, abs(y2 - y1))
            ratio = w / h
            gain = max(0.0, ratio - baseline)
            scores.append(min(1.0, gain / norm))
        return float(np.mean(scores)) if scores else 0.0

    def _box_horizontal_score(self, boxes):
        if not boxes:
            return 0.0
        scores = []
        for x1, y1, x2, y2 in boxes:
            w = max(1.0, abs(x2 - x1))
            h = max(1.0, abs(y2 - y1))
            score = max(0.0, (w - h) / max(h, 1.0))
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0

    def _fire_score(self, frame, person_boxes):
        logic = self.cfg["logic"]["fire"]
        if self.fire_mode == "yolo" and self.fire_model is not None:
            res = self.fire_model.predict(
                frame,
                imgsz=self.input_size,
                conf=logic.get("yolo_conf", 0.4),
                iou=logic.get("yolo_iou", 0.5),
                classes=self.fire_classes if self.fire_classes else None,
                half=(self.device == "cuda"),
                verbose=False
            )[0]
            if res.boxes is None or len(res.boxes) == 0:
                return 0.0
            max_conf = float(res.boxes.conf.max().cpu())
            return min(1.0, max_conf)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sat_min = logic.get("sat_min", 120)
            val_min = logic.get("val_min", 170)
            lower1 = np.array([0, sat_min, val_min], dtype=np.uint8)
            upper1 = np.array([25, 255, 255], dtype=np.uint8)
            lower2 = np.array([160, sat_min, val_min], dtype=np.uint8)
            upper2 = np.array([179, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

            median_sz = int(logic.get("median_blur", 3))
            if median_sz > 1:
                if median_sz % 2 == 0:
                    median_sz += 1
                mask = cv2.medianBlur(mask, median_sz)

            kernel_sz = int(logic.get("morph_kernel", 3))
            if kernel_sz > 1:
                kernel = np.ones((kernel_sz, kernel_sz), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            if person_boxes:
                ppl_mask = np.zeros(mask.shape, dtype=np.uint8)
                margin = int(logic.get("person_margin", 8))
                for x1, y1, x2, y2 in person_boxes:
                    x1n = max(0, int(x1) - margin)
                    y1n = max(0, int(y1) - margin)
                    x2n = min(mask.shape[1] - 1, int(x2) + margin)
                    y2n = min(mask.shape[0] - 1, int(y2) + margin)
                    cv2.rectangle(ppl_mask, (x1n, y1n), (x2n, y2n), 255, -1)
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(ppl_mask))

            fire_px = float(cv2.countNonZero(mask))
            ratio = fire_px / float(mask.size)
            min_ratio = float(logic.get("min_ratio", 0.0))
            if ratio < min_ratio:
                return 0.0
            boost = float(logic.get("boost", 1.0))
            return float(min(1.0, boost * ratio / max(1e-6, logic["hsv_ratio"])))

    def _fall_score(self, prev_centers, curr_centers, prev_boxes, curr_boxes, pose_stats):
        if not curr_boxes:
            return 0.0
        cfg = self.cfg["logic"]["fall"]
        
        # Box aspect ratio (width > height)
        box_scores = []
        for x1, y1, x2, y2 in curr_boxes:
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if h > 0 and w > h:
                box_scores.append(1.0)
            else:
                box_scores.append(0.0)
        
        box_fall = float(np.mean(box_scores)) if box_scores else 0.0
        
        # Pose-based: shoulders below hips/feet
        pose_fall = 0.0
        if pose_stats.get("count", 0) > 0:
            pose_fall = pose_stats.get("lying", 0.0)
        
        # Combine
        score = (
            cfg.get("w_box", 0.7) * box_fall +
            cfg.get("w_pose", 0.3) * pose_fall
        )
        return float(min(1.0, score))

    def _violence_score(self, centers, flow_mag=0.0):
        if len(centers) < 2:
            return 0.0
        cfg = self.cfg["logic"]["violence"]
        prox_thr = cfg["prox_px"]
        prox_scores = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                prox_scores.append(1.0 / max(1.0, d / prox_thr))
        base = np.mean(prox_scores) if prox_scores else 0.0
        flow_thr = cfg["flow_mag"]
        flow_score = min(1.0, flow_mag / max(1e-6, flow_thr))
        crowd_factor = min(1.0, len(centers) / float(cfg.get("crowd_count", 3)))
        return float(min(1.0, 0.5 * base + 0.3 * flow_score + 0.2 * crowd_factor))

    def _resp_score(self, pose_stats):
        if pose_stats["count"] == 0:
            return 0.0
        cfg = self.cfg["logic"]["resp"]
        chest_avg = max(0.0, 1.0 - pose_stats["hand_chest"])
        chest_min = max(0.0, 1.0 - pose_stats["hand_chest_min"])
        torso_flat = max(0.0, 1.0 - pose_stats["torso_vertical"])
        score = (
            cfg.get("w_hands", 0.5) * chest_avg +
            cfg.get("w_hands_close", 0.3) * chest_min +
            cfg.get("w_torso", 0.2) * torso_flat
        )
        return float(min(1.0, score))

    def _severe_injury_score(self, pose_stats, fall_score, motion_value):
        if pose_stats["count"] == 0:
            return 0.0
        cfg = self.cfg["logic"]["severe_injury"]
        lying = pose_stats["lying"]
        distress = max(0.0, 1.0 - pose_stats["hand_chest"])
        stillness = max(0.0, 1.0 - motion_value / max(1e-6, cfg["motion_eps"]))
        score = (
            cfg.get("w_lying", 0.5) * lying +
            cfg.get("w_distress", 0.25) * max(distress, fall_score) +
            cfg.get("w_still", 0.25) * stillness
        )
        if lying < cfg.get("lying_min", 0.4):
            score *= lying / max(cfg.get("lying_min", 0.4), 1e-6)
        return float(min(1.0, score))

    def process(self, source=0, display=True):
        if isinstance(source, str) and source.isdigit():
            src_handle = int(source)
        else:
            src_handle = source

        cap = cv2.VideoCapture(src_handle)
        if not cap.isOpened():
            print("[ERROR] cannot open source:", source)
            return
        print("[INFO] device:", self.device, "| YOLO on GPU?", self.device == "cuda")

        window_name = "VitalSight v2"
        seek = {"active": False, "pending": False, "target": 0, "max": 0}
        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames > 0 and not isinstance(src_handle, int):
                seek["active"] = True
                seek["max"] = max(total_frames - 1, 1)

                def _on_seek(val):
                    seek["pending"] = True
                    seek["target"] = val

                cv2.createTrackbar("Seek", window_name, 0, seek["max"], _on_seek)

        while True:
            if seek["active"] and seek["pending"]:
                target = max(0, min(seek["target"], seek["max"]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                self.track_history.clear()
                self.prev_person_boxes = []
                self.person_boxes = []
                seek["pending"] = False

            ok, frame = cap.read()
            if not ok:
                break

            orig = frame.copy()
            h_frame, w_frame = frame.shape[:2]

            frame_rs = self._preprocess(frame)
            res, person_boxes = self._detect_persons(frame_rs)

            centers = self._centers(person_boxes)
            self.track_history.append(centers)
            motion_value = self._motion_value()
            flow_mag = motion_value

            kp_map = {}
            pose_feats = []
            if self.pose_enabled and person_boxes:
                areas = [max(1.0, (x2 - x1) * (y2 - y1)) for (x1, y1, x2, y2) in person_boxes]
                idxs = np.argsort(areas)[::-1][:self.pose_max_people]
                min_aspect = self.cfg["pose"].get("min_aspect", 0.0)
                sel = []
                for i in idxs:
                    x1, y1, x2, y2 = person_boxes[i]
                    area_frac = areas[i] / float(self.input_size * self.input_size)
                    h_box = max(1.0, y2 - y1)
                    w_box = max(1.0, x2 - x1)
                    aspect = h_box / w_box
                    if area_frac < self.min_box_area_frac:
                        continue
                    if min_aspect > 0.0 and aspect < min_aspect:
                        continue
                    sel.append(person_boxes[i])
                if sel:
                    kp_map = self.pose_est.infer(frame_rs, sel)
                    for k in kp_map.values():
                        feats = keypose_feats(k)
                        pose_feats.append(feats)

            pose_stats = self._pose_stats(pose_feats)

            prev_centers = self.track_history[-2] if len(self.track_history) > 1 else []
            prev_boxes = self.prev_person_boxes

            fire_s = self._fire_score(frame_rs, person_boxes)
            fall_s = self._fall_score(prev_centers, centers, prev_boxes, person_boxes, pose_stats) if centers else 0.0
            resp_s = self._resp_score(pose_stats)
            viol_s = self._violence_score(centers, flow_mag) if centers else 0.0
            severe_s = self._severe_injury_score(pose_stats, fall_s, motion_value)

            scores = {
                "fire": min(1.0, fire_s),
                "fall": fall_s,
                "respiratory_distress": resp_s,
                "violence_panic": viol_s,
                "severe_injury": severe_s,
            }

            active = []
            for k, v in scores.items():
                is_on, sm = self.db[k].update(v)
                if is_on:
                    active.append((k, float(sm)))

            annotated = orig
            scale_x = w_frame / float(self.input_size)
            scale_y = h_frame / float(self.input_size)

            if res and hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else []
                for idx, bbox in enumerate(boxes_xyxy):
                    x1, y1, x2, y2 = bbox
                    x1o = int(round(x1 * scale_x))
                    y1o = int(round(y1 * scale_y))
                    x2o = int(round(x2 * scale_x))
                    y2o = int(round(y2 * scale_y))
                    cv2.rectangle(annotated, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
                    if idx < len(confs):
                        cv2.putText(
                            annotated,
                            f"person {confs[idx]:.2f}",
                            (x1o, max(0, y1o - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

            if kp_map:
                self._draw_pose(annotated, kp_map, scale_x, scale_y)

            y0 = 28
            label_text = " | ".join([f"{k}:{s:.2f}" for k, s in active]) if active else "none"
            color = (0, 0, 255) if active else (0, 255, 0)
            cv2.putText(annotated, label_text, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if self.cfg["runtime"]["show_fps"]:
                t = time.time()
                dt = t - self.last_fps_t
                self.last_fps_t = t
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-6, dt))
                cv2.putText(annotated, f"{self.fps:5.1f} FPS", (12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if display:
                cv2.imshow("VitalSight v2", annotated)
                if seek["active"] and not seek["pending"]:
                    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cv2.setTrackbarPos("Seek", window_name, min(seek["max"], pos))
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

        cap.release()
        if display:
            cv2.destroyWindow(window_name)
        cv2.destroyAllWindows()

    def _pose_stats(self, pose_feats):
        if not pose_feats:
            return {"count": 0, "lying": 0.0, "hand_chest": 1.0, "hand_chest_min": 1.0, "torso_vertical": 1.0}
        lying = np.mean([np.clip(f.get("lying_score", 0.0), 0.0, 1.0) for f in pose_feats])
        hand_vals = [np.clip(f.get("hand_chest", 1.0), 0.0, 1.0) for f in pose_feats]
        hand = np.mean(hand_vals)
        hand_min = np.min(hand_vals)
        torso = np.mean([np.clip(f.get("torso_angle", 1.0), 0.0, 1.0) for f in pose_feats])
        return {
            "count": len(pose_feats),
            "lying": float(lying),
            "hand_chest": float(hand),
            "hand_chest_min": float(hand_min),
            "torso_vertical": float(torso)
        }

    def _draw_pose(self, canvas, kp_map, scale_x, scale_y):
        h, w = canvas.shape[:2]
        for keypoints in kp_map.values():
            pts = np.asarray(keypoints, dtype=np.float32)
            pts_px = []
            for p in pts:
                if p.shape[0] < 2 or np.isnan(p[:2]).any():
                    pts_px.append(None)
                    continue
                x = int(np.clip(p[0], 0.0, 1.0) * self.input_size * scale_x)
                y = int(np.clip(p[1], 0.0, 1.0) * self.input_size * scale_y)
                if 0 <= x < w and 0 <= y < h:
                    pts_px.append((x, y))
                else:
                    pts_px.append(None)

            for a, b in POSE_SKELETON:
                if a < len(pts_px) and b < len(pts_px):
                    pa, pb = pts_px[a], pts_px[b]
                    if pa is None or pb is None:
                        continue
                    cv2.line(canvas, pa, pb, (0, 200, 255), 2)

            for pt in pts_px:
                if pt is None:
                    continue
                cv2.circle(canvas, pt, 3, (0, 255, 255), -1)
