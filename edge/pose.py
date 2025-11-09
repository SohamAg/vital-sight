# edge/pose.py
import numpy as np

YOLO_BACKENDS = {"yolo11n-pose", "yolov8n-pose"}  # support both names

class PoseEstimator:
    def __init__(self, backend="yolo11n-pose"):
        self.backend = backend
        self.impl = None
        self.pose = None

        if backend in YOLO_BACKENDS:
            from ultralytics import YOLO
            weights = "yolo11n-pose.pt" if "11" in backend else "yolov8n-pose.pt"
            self.impl = YOLO(weights)
        elif backend == "mediapipe":
            try:
                import mediapipe as mp
            except ImportError as e:
                raise RuntimeError("mediapipe not installed. pip install mediapipe") from e
            self.mp = mp
            self.pose = mp.solutions.pose.Pose(model_complexity=0)
        else:
            raise ValueError(f"Unsupported pose backend: {backend}")

    def infer(self, frame_bgr, person_boxes_xyxy):
        """
        Returns dict: local_idx -> keypoints ndarray (num_kpts,2) normalized [0..1]
        local_idx corresponds to the order of person_boxes_xyxy given.
        """
        h, w, _ = frame_bgr.shape
        out = {}
        if self.backend in YOLO_BACKENDS:
            # Run once on full frame, then greedily match to input boxes by center proximity
            res = self.impl(frame_bgr, verbose=False)[0]
            if res.keypoints is None or len(res.keypoints.xyn) == 0:
                return out
            kps = res.keypoints.xyn.cpu().numpy()  # (P, K, 2) in [0..1]
            # Build centers for supplied boxes (normalized)
            box_cs = [(((x1+x2)/2)/w, ((y1+y2)/2)/h) for (x1,y1,x2,y2) in person_boxes_xyxy]
            # Build centers for YOLO pose persons (from keypoints)
            person_cs = []
            for i in range(len(kps)):
                kp = kps[i]
                # avg of visible keypoints
                cx, cy = float(np.mean(kp[:,0])), float(np.mean(kp[:,1]))
                person_cs.append((cx,cy))
            # Greedy nearest matching
            used = set()
            for i, bc in enumerate(box_cs):
                if not person_cs: break
                j = int(np.argmin([np.hypot(bc[0]-pc[0], bc[1]-pc[1]) for pc in person_cs]))
                if j in used: continue
                out[i] = kps[j][:,:2]
                used.add(j)
            return out

        elif self.backend == "mediapipe":
            if self.pose is None:
                raise RuntimeError("MediaPipe pose not initialized")
            for i,(x1,y1,x2,y2) in enumerate(person_boxes_xyxy):
                x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
                crop = frame_bgr[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if crop.size == 0: continue
                rgb = crop[:,:,::-1]
                res = self.pose.process(rgb)
                if not res or not res.pose_landmarks: continue
                pts = []
                for lm in res.pose_landmarks.landmark:
                    px = (x1 + lm.x * (x2-x1)) / w
                    py = (y1 + lm.y * (y2-y1)) / h
                    pts.append([px,py])
                out[i] = np.array(pts, dtype=np.float32)
            return out

        else:
            return out

def keypose_feats(keypoints):
    """Compute simple pose features for fall/immobility/resp hints."""
    def safe(k, idx): 
        return k[idx] if (k is not None and idx < len(k)) else None

    L_SH, R_SH = safe(keypoints, 11), safe(keypoints, 12)
    L_HP, R_HP = safe(keypoints, 23), safe(keypoints, 24)
    L_WR, R_WR = safe(keypoints, 15), safe(keypoints, 16)
    feats = {"lying_score":0.0, "hand_chest":1.0, "torso_angle":0.0}

    if all(v is not None for v in [L_SH, R_SH, L_HP, R_HP]):
        sh = np.mean([L_SH, R_SH], axis=0)
        hp = np.mean([L_HP, R_HP], axis=0)
        torso = hp - sh
        angle = np.arctan2(torso[1], torso[0])  # radians
        feats["torso_angle"] = float(abs(np.cos(angle)))  # ~1 vertical, ~0 horizontal
        feats["lying_score"] = 1.0 - feats["torso_angle"]

        chest = sh*0.6 + hp*0.4
        wrs = [p for p in [L_WR, R_WR] if p is not None]
        if wrs:
            d = min(np.linalg.norm(w - chest) for w in wrs)
            feats["hand_chest"] = float(d)  # normalized 0..1
    return feats
