from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import numpy as np
from scipy.signal import savgol_filter
import math
import os
import base64
import urllib.request
import threading
import time
from collections import deque

app = Flask(__name__, static_folder='static', template_folder='.')
app.config['SECRET_KEY'] = 'posture_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading pose model (~25 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model ready.")
    else:
        print(f"[INFO] Model found: {MODEL_PATH}")

# ─────────────────────────────────────────────────────────────
# MEDIAPIPE CONNECTIONS
# ─────────────────────────────────────────────────────────────

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(17,19),
    (12,14),(14,16),(16,18),(16,20),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

# These are only for frontend display cards and general body analysis
JOINT_ANGLES = {
    "L Shoulder":   (13, 11, 23),
    "R Shoulder":   (14, 12, 24),
    "L Elbow":      (11, 13, 15),
    "R Elbow":      (12, 14, 16),
    "L Wrist":      (13, 15, 19),
    "R Wrist":      (14, 16, 20),
    "L Hip":        (11, 23, 25),
    "R Hip":        (12, 24, 26),
    "L Knee":       (23, 25, 27),
    "R Knee":       (24, 26, 28),
    "L Ankle":      (25, 27, 29),
    "R Ankle":      (26, 28, 30),
    "L Foot":       (27, 29, 31),
    "R Foot":       (28, 30, 32),
    "Pelvis":       (23, 24, 26),
    "Torso":        (23, 11, 12),
    "Neck":         (7, 0, 8),
    "Spine":        (11, 23, 25),
    "L Neck Side":  (7, 11, 13),
    "R Neck Side":  (8, 12, 14),
}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def safe_point(landmarks, idx):
    if idx < 0 or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    return np.array([lm["x"], lm["y"]], dtype=float)

def safe_vis(landmarks, idx):
    if idx < 0 or idx >= len(landmarks):
        return 0.0
    return float(landmarks[idx].get("vis", 0.0))

def midpoint(a, b):
    return (a + b) / 2.0

def distance(a, b):
    return float(np.linalg.norm(a - b))

def angle_abc(a, b, c):
    ba = a - b
    bc = c - b
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_ang = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))

def angle_of_vector_from_vertical(p1, p2):
    """
    0° means perfectly vertical.
    Larger means more tilted.
    """
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return 0.0
    vertical = np.array([0.0, -1.0], dtype=float)
    unit = vec / norm
    cos_ang = np.clip(np.dot(unit, vertical), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def avg_visibility(landmarks, ids):
    vals = [safe_vis(landmarks, i) for i in ids]
    return sum(vals) / len(vals) if vals else 0.0

def visibility_ok(landmarks, ids, min_vis=0.45):
    return all(safe_vis(landmarks, i) >= min_vis for i in ids)

def side_label(name):
    return "left" if name.startswith("L ") else "right" if name.startswith("R ") else "center"

# ─────────────────────────────────────────────────────────────
# IMPROVED POSTURE ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyze_posture(angles, landmarks):
    issues = []
    score = 100

    if not landmarks or len(landmarks) < 33:
        return {
            "issues": [{
                "joint": "Detection",
                "message": "Insufficient landmarks for posture analysis",
                "severity": "medium",
                "angle": 0,
                "ideal": "Stable full-body detection"
            }],
            "score": 0,
            "status": "poor",
            "summary": "Pose not detected clearly."
        }

    def pt(i):
        return np.array([landmarks[i]["x"], landmarks[i]["y"]], dtype=float)

    def zval(i):
        return float(landmarks[i]["z"])

    def vis(i):
        return float(landmarks[i]["vis"])

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def severity_from_ratio(v, mild, strong):
        if v > strong:
            return "high"
        elif v > mild:
            return "medium"
        return None

    key_ids = [0, 11, 12, 23, 24]
    if any(vis(i) < 0.40 for i in key_ids):
        return {
            "issues": [{
                "joint": "Detection",
                "message": "Main body points are not visible enough",
                "severity": "medium",
                "angle": 0,
                "ideal": "Front-facing, well-lit body"
            }],
            "score": 35,
            "status": "poor",
            "summary": "Low confidence pose. Improve camera view."
        }

    nose = pt(0)
    l_sh = pt(11)
    r_sh = pt(12)
    l_hip = pt(23)
    r_hip = pt(24)

    sh_center = (l_sh + r_sh) / 2.0
    hip_center = (l_hip + r_hip) / 2.0

    torso_len = dist(sh_center, hip_center)
    if torso_len < 1e-5:
        torso_len = 0.2

    shoulder_width = dist(l_sh, r_sh)
    if shoulder_width < 1e-5:
        shoulder_width = 0.15

    # -----------------------------
    # 1. Forward head detection
    # -----------------------------
    head_forward_x = abs(nose[0] - sh_center[0]) / shoulder_width
    sev = severity_from_ratio(head_forward_x, 0.18, 0.30)
    if sev:
        issues.append({
            "joint": "Neck",
            "message": "Head is drifting forward relative to the shoulders",
            "severity": sev,
            "angle": round(head_forward_x * 100, 1),
            "ideal": "< 18% shoulder-width offset"
        })
        score -= 10 if sev == "medium" else 20

    # -----------------------------
    # 2. Neck angle from your angle map
    # better thresholds
    # -----------------------------
    neck = float(angles.get("Neck", 0))
    if neck > 0 and neck < 150:
        sev = "high" if neck < 135 else "medium"
        issues.append({
            "joint": "Neck",
            "message": "Forward neck posture detected — bring head back over shoulders",
            "severity": sev,
            "angle": round(neck, 1),
            "ideal": "150–180°"
        })
        score -= 10 if sev == "medium" else 18

    # -----------------------------
    # 3. Spine / torso bend
    # -----------------------------
    spine = float(angles.get("Spine", 0))
    if spine > 0 and spine < 158:
        sev = "high" if spine < 145 else "medium"
        issues.append({
            "joint": "Spine",
            "message": "Torso is bent or slouched — straighten your upper body",
            "severity": sev,
            "angle": round(spine, 1),
            "ideal": "158–180°"
        })
        score -= 10 if sev == "medium" else 18

    # -----------------------------
    # 4. Shoulder collapse / rounding
    # if shoulders move closer vertically to hips,
    # torso gets visually compressed
    # -----------------------------
    torso_height_ratio = abs(sh_center[1] - hip_center[1])
    # smaller height ratio means upper body collapsed
    # normalized by torso_len from current frame
    compression = torso_height_ratio / torso_len if torso_len > 1e-5 else 1.0

    # for normal standing/sitting, compression stays near 1
    # if person slouches heavily, this drops
    if compression < 0.88:
        sev = "high" if compression < 0.78 else "medium"
        issues.append({
            "joint": "Upper Body",
            "message": "Torso looks compressed — chest may be collapsing downward",
            "severity": sev,
            "angle": round(compression * 100, 1),
            "ideal": "> 88%"
        })
        score -= 8 if sev == "medium" else 15

    # -----------------------------
    # 5. Shoulder rounding using z depth
    # mediapipe z: more negative = closer to camera
    # if shoulders come forward a lot relative to hips/head, slouch likely
    # -----------------------------
    shoulder_z = (zval(11) + zval(12)) / 2.0
    hip_z = (zval(23) + zval(24)) / 2.0
    nose_z = zval(0)

    shoulder_forward = shoulder_z - hip_z
    head_forward_z = nose_z - shoulder_z

    # Rounded upper body / shoulders
    if shoulder_forward < -0.08:
        sev = "high" if shoulder_forward < -0.14 else "medium"
        issues.append({
            "joint": "Shoulders",
            "message": "Shoulders appear rounded forward",
            "severity": sev,
            "angle": round(abs(shoulder_forward) * 100, 1),
            "ideal": "Shoulders not far ahead of hips"
        })
        score -= 8 if sev == "medium" else 14

    # Head jutting ahead in depth
    if head_forward_z < -0.05:
        sev = "high" if head_forward_z < -0.10 else "medium"
        issues.append({
            "joint": "Head Position",
            "message": "Head is projecting forward in front of the torso",
            "severity": sev,
            "angle": round(abs(head_forward_z) * 100, 1),
            "ideal": "Head aligned over shoulders"
        })
        score -= 8 if sev == "medium" else 14

    # -----------------------------
    # 6. Shoulder symmetry
    # -----------------------------
    l_sh_ang = float(angles.get("L Shoulder", 0))
    r_sh_ang = float(angles.get("R Shoulder", 0))
    if l_sh_ang > 0 and r_sh_ang > 0:
        sh_diff = abs(l_sh_ang - r_sh_ang)
        if sh_diff > 15:
            issues.append({
                "joint": "Shoulders",
                "message": f"Uneven shoulder angles (L:{l_sh_ang:.0f}° vs R:{r_sh_ang:.0f}°)",
                "severity": "medium",
                "angle": round(sh_diff, 1),
                "ideal": "< 15° difference"
            })
            score -= 6

    # -----------------------------
    # 7. Hip symmetry
    # -----------------------------
    l_hip_ang = float(angles.get("L Hip", 0))
    r_hip_ang = float(angles.get("R Hip", 0))
    if l_hip_ang > 0 and r_hip_ang > 0:
        hip_diff = abs(l_hip_ang - r_hip_ang)
        if hip_diff > 15:
            issues.append({
                "joint": "Hips",
                "message": f"Uneven hip angles (L:{l_hip_ang:.0f}° vs R:{r_hip_ang:.0f}°)",
                "severity": "medium",
                "angle": round(hip_diff, 1),
                "ideal": "< 15° difference"
            })
            score -= 6

    # -----------------------------
    # 8. Combined slouch rule
    # this is the important part
    # -----------------------------
    slouch_flags = 0
    if neck > 0 and neck < 150:
        slouch_flags += 1
    if spine > 0 and spine < 158:
        slouch_flags += 1
    if head_forward_x > 0.18:
        slouch_flags += 1
    if shoulder_forward < -0.08:
        slouch_flags += 1
    if head_forward_z < -0.05:
        slouch_flags += 1

    if slouch_flags >= 3:
        issues.append({
            "joint": "Slouching",
            "message": "Clear slouching pattern detected — lift chest and align head over torso",
            "severity": "high",
            "angle": slouch_flags,
            "ideal": "No slouch indicators"
        })
        score -= 18
    elif slouch_flags == 2:
        issues.append({
            "joint": "Slouching",
            "message": "Mild slouching pattern detected",
            "severity": "medium",
            "angle": slouch_flags,
            "ideal": "No slouch indicators"
        })
        score -= 10

    score = int(round(clamp(score, 0, 100)))

    if score >= 88:
        status = "excellent"
        summary = "Great posture! Body alignment looks balanced."
    elif score >= 72:
        status = "good"
        summary = "Generally good posture with small corrections needed."
    elif score >= 52:
        status = "fair"
        summary = "Posture issues detected. Try resetting to upright neutral."
    else:
        status = "poor"
        summary = "Poor posture detected. Sit or stand taller and realign your head and shoulders."

    return {
        "issues": issues,
        "score": score,
        "status": status,
        "summary": summary
    }

# ─────────────────────────────────────────────────────────────
# CAMERA THREAD
# ─────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.cap = None
        self.landmarker = None
        self.smooth_window = 7
        self.smooth_poly = 3
        self.hist = [deque(maxlen=30) for _ in range(33)]
        self.ang_smooth = {}
        self.prev_emit_time = time.time()

    def start_camera(self):
        ensure_model()

        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_segmentation_masks=False,
        )

        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("[ERROR] Could not open camera.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.running = True
        self.start()
        return True

    def stop_camera(self):
        self.running = False
        time.sleep(0.3)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def smooth_angle(self, name, val, alpha=0.25):
        if name not in self.ang_smooth:
            self.ang_smooth[name] = val
        else:
            self.ang_smooth[name] = alpha * val + (1 - alpha) * self.ang_smooth[name]
        return self.ang_smooth[name]

    def smooth_landmarks(self, pts):
        for i, pt in enumerate(pts):
            self.hist[i].append(pt)

        out = []
        for i, pt in enumerate(pts):
            h = np.array(self.hist[i])
            if len(h) >= self.smooth_window:
                try:
                    xs = savgol_filter(h[:, 0], self.smooth_window, self.smooth_poly)
                    ys = savgol_filter(h[:, 1], self.smooth_window, self.smooth_poly)
                    out.append(np.array([xs[-1], ys[-1]], dtype=float))
                except Exception:
                    out.append(pt.copy())
            else:
                out.append(pt.copy())
        return out

    def calc_angle(self, a, b, c):
        return angle_abc(a, b, c)

    def draw_skeleton(self, frame, pts, vis):
        # connections
        for a, b in POSE_CONNECTIONS:
            if a >= len(pts) or b >= len(pts):
                continue
            if vis[a] < 0.35 or vis[b] < 0.35:
                continue

            p1 = tuple(pts[a].astype(int))
            p2 = tuple(pts[b].astype(int))
            cv2.line(frame, p1, p2, (90, 220, 255), 2, cv2.LINE_AA)

        # joints
        for i, pt in enumerate(pts):
            if i >= len(vis) or vis[i] < 0.35:
                continue
            p = tuple(pt.astype(int))
            cv2.circle(frame, p, 5, (0, 255, 180), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 7, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_status_text(self, frame, posture):
        if posture is None:
            return

        score = posture.get("score", 0)
        status = posture.get("status", "unknown").upper()
        summary = posture.get("summary", "")

        color = (0, 220, 120)
        if score < 55:
            color = (60, 60, 255)
        elif score < 74:
            color = (0, 180, 255)
        elif score < 88:
            color = (255, 180, 0)

        cv2.rectangle(frame, (14, 14), (360, 110), (15, 20, 30), -1)
        cv2.rectangle(frame, (14, 14), (360, 110), color, 2)

        cv2.putText(frame, f"POSTURE: {status}", (28, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"SCORE: {score}", (28, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, summary[:44], (28, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 235), 1, cv2.LINE_AA)

    def run(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror view
            H, W = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)

            try:
                result = self.landmarker.detect_for_video(mp_img, timestamp_ms)
            except Exception as e:
                print(f"[WARN] Detection error: {e}")
                continue

            pose_data = {
                "detected": False,
                "angles": {},
                "posture": None,
                "landmarks": [],
                "connections": POSE_CONNECTIONS
            }

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                pts = np.array([[lm.x * W, lm.y * H] for lm in lms], dtype=float)
                vis = [float(lm.visibility) for lm in lms]

                smoothed = self.smooth_landmarks(pts)

                angles = {}
                for name, (ai, bi, ci) in JOINT_ANGLES.items():
                    if max(ai, bi, ci) < len(smoothed):
                        raw = self.calc_angle(smoothed[ai], smoothed[bi], smoothed[ci])
                        angles[name] = round(self.smooth_angle(name, raw), 1)

                self.draw_skeleton(frame, smoothed, vis)

                lm_norm = [
                    {
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "vis": float(lm.visibility)
                    }
                    for lm in lms
                ]

                posture = analyze_posture(angles, lm_norm)
                self.draw_status_text(frame, posture)

                pose_data = {
                    "detected": True,
                    "angles": angles,
                    "posture": posture,
                    "landmarks": lm_norm,
                    "connections": POSE_CONNECTIONS
                }

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            b64 = base64.b64encode(buf).decode("utf-8")
            pose_data["frame"] = b64

            socketio.emit("frame", pose_data)
            time.sleep(0.033)

        self.running = False

# ─────────────────────────────────────────────────────────────
# APP ROUTES
# ─────────────────────────────────────────────────────────────

camera_thread = CameraThread()

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@socketio.on("start_camera")
def handle_start():
    global camera_thread
    if not camera_thread.running:
        camera_thread = CameraThread()
        success = camera_thread.start_camera()
        emit("camera_status", {"status": "started" if success else "error"})
    else:
        emit("camera_status", {"status": "already_running"})

@socketio.on("stop_camera")
def handle_stop():
    global camera_thread
    if camera_thread.running:
        camera_thread.stop_camera()
        emit("camera_status", {"status": "stopped"})
    else:
        emit("camera_status", {"status": "already_stopped"})

@socketio.on("disconnect")
def handle_disconnect():
    # do not force stop for every disconnect if you expect multiple tabs
    pass

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  POSTURE ANALYSIS SERVER")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
