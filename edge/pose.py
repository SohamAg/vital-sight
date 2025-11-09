# edge/pose.py
import numpy as np

class PoseEstimator:
    def __init__(self, backend="yolov8n-pose"):
        self.backend = backend
        self.impl = None
        if backend == "yolov8n-pose":
            from ultralytics import YOLO
            self.impl = YOLO("yolov8n-pose.pt")
        elif backend == "mediapipe":
            import mediapipe as mp
            self.mp = mp
            self.pose = mp.solutions.pose.Pose(model_complexity=0)
        else:
            raise ValueError("Unsupported pose backend")

    def infer(self, frame_bgr, person_boxes_xyxy):
        """
        Returns dict: track_idx -> keypoints ndarray shape (N,2) normalized [0..1]
        Here track_idx is just index into person_boxes order for simplicity.
        """
        h, w, _ = frame_bgr.shape
        out = {}
        if self.backend == "yolov8n-pose":
            # Run on full frame, then match to boxes (fast enough for max_people=2)
            res = self.impl(frame_bgr, verbose=False)[0]
            if res.keypoints is None: return out
            kps = res.keypoints.xyn.cpu().numpy()  # (num_people, num_kpts, 2) in [0..1]
            # naive matching: nearest center
            centers = [((x1+x2)/2,(y1+y2)/2) for (x1,y1,x2,y2) in person_boxes_xyxy]
            for idx, kp in enumerate(kps):
                if idx >= len(centers): break
                out[idx] = kp[:, :2]  # (num_kpts,2)
            return out
        else:
            # mediapipe: run per-crop (good for webcam close-up)
            for i,(x1,y1,x2,y2) in enumerate(person_boxes_xyxy):
                x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
                crop = frame_bgr[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if crop.size == 0: continue
                rgb = crop[:,:,::-1]
                res = self.pose.process(rgb)
                if not res.pose_landmarks: continue
                pts = []
                for lm in res.pose_landmarks.landmark:
                    # normalize back to full-frame coordinates
                    px = (x1 + lm.x * (x2-x1)) / w
                    py = (y1 + lm.y * (y2-y1)) / h
                    pts.append([px,py])
                out[i] = np.array(pts, dtype=np.float32)
            return out

def keypose_feats(keypoints):
    """
    Compute simple pose features for fall/immobility/resp hints.
    expects keypoints normalized [0..1] in image coords.
    Returns dict with lying_score, hand_chest_dist (px normalized), torso_angle
    """
    # COCO-ish index mapping is model-specific; we’ll be conservative
    # Use shoulders (11,12), hips (23,24), wrists (15,16) if present
    def safe(idx): 
        return keypoints[idx] if (keypoints is not None and idx < len(keypoints)) else None
    L_SH, R_SH = safe(11), safe(12)
    L_HP, R_HP = safe(23), safe(24)
    L_WR, R_WR = safe(15), safe(16)
    feats = {"lying_score":0.0, "hand_chest":1.0, "torso_angle":0.0}
    if L_SH is not None and R_SH is not None and L_HP is not None and R_HP is not None:
        sh = np.mean([L_SH, R_SH], axis=0)
        hp = np.mean([L_HP, R_HP], axis=0)
        torso = hp - sh
        # angle vs vertical
        angle = np.arctan2(torso[1], torso[0])  # radians
        feats["torso_angle"] = float(abs(np.cos(angle)))  # ~0 horizontal, ~1 vertical
        # lying if torso is more horizontal
        feats["lying_score"] = 1.0 - feats["torso_angle"]

        # chest center approx midway between shoulders and hips (upper torso)
        chest = sh*0.6 + hp*0.4
        wrs = [p for p in [L_WR, R_WR] if p is not None]
        if wrs:
            d = min(np.linalg.norm(w - chest) for w in wrs)
            feats["hand_chest"] = float(d)  # normalized 0..1
    return feats
