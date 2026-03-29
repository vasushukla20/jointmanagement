"""
═══════════════════════════════════════════════════════════════════════════════
  POSTURE ANALYSIS WEB SERVER
  Flask + SocketIO + MediaPipe
  
  Install:
    pip install flask flask-socketio flask-cors mediapipe opencv-python scipy
  
  Run:
    python server.py
  
  Then open: http://localhost:5000
═══════════════════════════════════════════════════════════════════════════════
"""

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
import math, os, base64, urllib.request, threading, time
from collections import deque

# ─── APP SETUP ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static', template_folder='.')
app.config['SECRET_KEY'] = 'posture_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── MODEL ───────────────────────────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading pose model (~25 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model ready.")
    else:
        print(f"[INFO] Model found: {MODEL_PATH}")

# ─── JOINT CONFIG ────────────────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(17,19),
    (12,14),(14,16),(16,18),(16,20),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

JOINT_ANGLES = {
    "L Shoulder": (13, 11, 23),
    "R Shoulder": (14, 12, 24),
    "L Elbow":    (11, 13, 15),
    "R Elbow":    (12, 14, 16),
    "L Hip":      (11, 23, 25),
    "R Hip":      (12, 24, 26),
    "L Knee":     (23, 25, 27),
    "R Knee":     (24, 26, 28),
    "L Ankle":    (25, 27, 29),
    "R Ankle":    (26, 28, 30),
    "Neck":       (11, 0, 12),
    "Spine":      (0, 11, 23),
}

# ─── POSTURE RULES ───────────────────────────────────────────────────────────
def analyze_posture(angles, landmarks):
    """
    Analyze posture based on joint angles and landmark positions.
    Returns: list of issues, overall score (0-100), status
    """
    issues = []
    score = 100

    def get_angle(name):
        return angles.get(name, 0)

    # 1. Neck / Head forward
    neck = get_angle("Neck")
    if neck < 140:
        issues.append({
            "joint": "Neck",
            "message": "Head is tilting forward (forward head posture)",
            "severity": "high" if neck < 120 else "medium",
            "angle": round(neck, 1),
            "ideal": "160–180°"
        })
        score -= 20 if neck < 120 else 10

    # 2. Spine alignment
    spine = get_angle("Spine")
    if spine < 150:
        issues.append({
            "joint": "Spine",
            "message": "Spine is curved — sit/stand straighter",
            "severity": "high" if spine < 130 else "medium",
            "angle": round(spine, 1),
            "ideal": "165–180°"
        })
        score -= 20 if spine < 130 else 10

    # 3. Shoulder symmetry
    ls = get_angle("L Shoulder")
    rs = get_angle("R Shoulder")
    if abs(ls - rs) > 15:
        issues.append({
            "joint": "Shoulders",
            "message": f"Uneven shoulders (L:{ls:.0f}° vs R:{rs:.0f}°) — possible tilt",
            "severity": "medium",
            "angle": round(abs(ls - rs), 1),
            "ideal": "< 10° difference"
        })
        score -= 10

    # 4. Hip symmetry
    lh = get_angle("L Hip")
    rh = get_angle("R Hip")
    if abs(lh - rh) > 15:
        issues.append({
            "joint": "Hips",
            "message": f"Uneven hip alignment (L:{lh:.0f}° vs R:{rh:.0f}°)",
            "severity": "medium",
            "angle": round(abs(lh - rh), 1),
            "ideal": "< 10° difference"
        })
        score -= 10

    # 5. Knee hyperextension check
    for side in ["L", "R"]:
        knee = get_angle(f"{side} Knee")
        if knee > 185:
            issues.append({
                "joint": f"{side} Knee",
                "message": f"{side} knee may be hyperextended",
                "severity": "medium",
                "angle": round(knee, 1),
                "ideal": "170–185°"
            })
            score -= 8

    # 6. Slouching (shoulder angle very low)
    avg_shoulder = (ls + rs) / 2
    if avg_shoulder < 60:
        issues.append({
            "joint": "Upper Body",
            "message": "Severe slouching detected — raise your chest",
            "severity": "high",
            "angle": round(avg_shoulder, 1),
            "ideal": "> 80°"
        })
        score -= 15

    score = max(0, min(100, score))

    if score >= 85:
        status = "excellent"
        summary = "Great posture! Keep it up."
    elif score >= 65:
        status = "good"
        summary = "Decent posture with minor issues."
    elif score >= 40:
        status = "fair"
        summary = "Several posture issues detected."
    else:
        status = "poor"
        summary = "Poor posture — please correct immediately."

    return {
        "issues": issues,
        "score": round(score),
        "status": status,
        "summary": summary
    }


# ─── CAMERA THREAD ───────────────────────────────────────────────────────────
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

    def start_camera(self):
        ensure_model()
        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self.start()
        return True

    def stop_camera(self):
        self.running = False
        time.sleep(0.3)
        if self.cap:
            self.cap.release()
        if self.landmarker:
            self.landmarker.close()

    def smooth_angle(self, name, val, alpha=0.25):
        self.ang_smooth[name] = val if name not in self.ang_smooth \
            else alpha * val + (1 - alpha) * self.ang_smooth[name]
        return self.ang_smooth[name]

    def smooth_landmarks(self, pts):
        for i, pt in enumerate(pts):
            self.hist[i].append(pt)
        out = []
        for i, pt in enumerate(pts):
            h = np.array(self.hist[i])
            if len(h) >= self.smooth_window:
                out.append(np.array([
                    savgol_filter(h[:,0], self.smooth_window, self.smooth_poly)[-1],
                    savgol_filter(h[:,1], self.smooth_window, self.smooth_poly)[-1]
                ]))
            else:
                out.append(pt.copy())
        return out

    def calc_angle(self, a, b, c):
        ba = a - b; bc = c - b
        n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
        if n1 < 1e-6 or n2 < 1e-6: return 0.0
        return math.degrees(math.acos(np.clip(np.dot(ba,bc)/(n1*n2),-1.0,1.0)))

    def draw_skeleton(self, frame, pts, vis, W, H):
        for a, b in POSE_CONNECTIONS:
            if a >= len(pts) or b >= len(pts): continue
            if vis[a] < 0.4 or vis[b] < 0.4: continue
            p1 = tuple(pts[a].astype(int))
            p2 = tuple(pts[b].astype(int))
            cv2.line(frame, p1, p2, (100, 220, 255), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            if i >= len(vis) or vis[i] < 0.4: continue
            p = tuple(pt.astype(int))
            cv2.circle(frame, p, 5, (0, 255, 180), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 7, (255, 255, 255), 1, cv2.LINE_AA)

    def run(self):
        frame_idx = 0
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))

            try:
                result = self.landmarker.detect_for_video(mp_img, timestamp)
            except Exception as e:
                print(f"[WARN] Detection error: {e}")
                continue

            pose_data = {"detected": False, "angles": {}, "posture": None, "landmarks": []}

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                pts = np.array([[lm.x * W, lm.y * H] for lm in lms])
                vis = [lm.visibility for lm in lms]
                smoothed = self.smooth_landmarks(pts)

                angles = {}
                for jname, (ai, bi, ci) in JOINT_ANGLES.items():
                    if max(ai, bi, ci) < len(smoothed):
                        raw = self.calc_angle(smoothed[ai], smoothed[bi], smoothed[ci])
                        angles[jname] = round(self.smooth_angle(jname, raw), 1)

                self.draw_skeleton(frame, smoothed, vis, W, H)

                # Landmarks normalized for frontend canvas
                lm_norm = [{"x": lm.x, "y": lm.y, "z": lm.z, "vis": lm.visibility}
                           for lm in lms]

                posture = analyze_posture(angles, lm_norm)
                pose_data = {
                    "detected": True,
                    "angles": angles,
                    "posture": posture,
                    "landmarks": lm_norm,
                    "connections": POSE_CONNECTIONS
                }

            # Encode frame as JPEG base64
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode('utf-8')
            pose_data["frame"] = b64

            socketio.emit('frame', pose_data)
            frame_idx += 1
            time.sleep(0.033)  # ~30fps


camera_thread = CameraThread()

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# ─── SOCKET EVENTS ───────────────────────────────────────────────────────────
@socketio.on('start_camera')
def handle_start():
    global camera_thread
    if not camera_thread.running:
        camera_thread = CameraThread()
        success = camera_thread.start_camera()
        emit('camera_status', {'status': 'started' if success else 'error'})
    else:
        emit('camera_status', {'status': 'already_running'})

@socketio.on('stop_camera')
def handle_stop():
    global camera_thread
    if camera_thread.running:
        camera_thread.stop_camera()
        emit('camera_status', {'status': 'stopped'})

@socketio.on('disconnect')
def handle_disconnect():
    global camera_thread
    if camera_thread.running:
        camera_thread.stop_camera()

# ─── RUN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  POSTURE ANALYSIS SERVER")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)