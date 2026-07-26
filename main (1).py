"""
Face-recognition attendance system
───────────────────────────────────
Detector  : face_recognition HOG (dlib) — fast CPU-friendly detector
Liveness  : blink-only challenge (EAR below threshold for N consecutive frames)
Encoding  : skipped for faces already verified in the current live-session
"""

import atexit
import csv
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from flask import Flask, Response, jsonify, render_template

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FACES_FOLDER = BASE_DIR / "faces"
ATTENDANCE_FOLDER = BASE_DIR / "attendance"

# ---------------------------------------------------------------------------
# Camera / stream
# ---------------------------------------------------------------------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
CAMERA_GRAB_SKIP = 1
STREAM_SLEEP_SECONDS = 0.03
JPEG_QUALITY = 80
FRAME_STALE_SECONDS = 5.0

# ---------------------------------------------------------------------------
# Detection / recognition
# ---------------------------------------------------------------------------
CENTER_CROP_RATIO = 0.80          # slightly wider crop — HOG handles it fine
DETECTION_SCALE = 0.45            # raised from 0.35; HOG is cheap enough
ANALYSIS_INTERVAL_SECONDS = 0.20  # faster tick now that RetinaFace is gone
HOG_UPSAMPLE = 1                  # 0 = fastest; 1 = catches smaller faces
FACE_MATCH_TOLERANCE = 0.50

# ---------------------------------------------------------------------------
# Liveness — blink only
# ---------------------------------------------------------------------------
BLINK_EAR_THRESHOLD = 0.24        # EAR below this → eye is closed
BLINK_CONSECUTIVE_FRAMES = 2      # need this many closed frames in a row
LIVE_SESSION_SECONDS = 10         # skip re-challenge for this long after verification
CHALLENGE_TIMEOUT_SECONDS = 20    # reset challenge if face disappears this long

# ---------------------------------------------------------------------------
# Draw constants
# ---------------------------------------------------------------------------
DRAW_FONT = cv2.FONT_HERSHEY_SIMPLEX
DRAW_FONT_SCALE = 0.7
DRAW_FONT_THICKNESS = 2
DRAW_LABEL_OFFSET_X = 6
DRAW_LABEL_OFFSET_Y = 6
DRAW_LABEL_BAR_HEIGHT = 35
DRAW_BOX_THICKNESS = 2
DRAW_CROP_BOX_COLOR = (255, 255, 0)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_LIVE = (0, 255, 0)
COLOR_PENDING = (0, 165, 255)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
known_encodings: list = []
known_names: list = []

liveness_state: dict = {}

_attendance_lock = threading.Lock()
_attendance_cache_file: Path | None = None
_attendance_marked_names: set = set()

_latest_frame: tuple[np.ndarray, float] | None = None
_latest_results: list = []
_frame_lock = threading.Lock()
_results_lock = threading.Lock()

_camera: cv2.VideoCapture | None = None


# ---------------------------------------------------------------------------
# Known-face loading  (HOG-based encoding at enrolment)
# ---------------------------------------------------------------------------

def load_known_faces() -> None:
    if not FACES_FOLDER.exists():
        raise FileNotFoundError(f"Faces folder not found: {FACES_FOLDER}")

    for image_path in FACES_FOLDER.iterdir():
        if image_path.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
            continue

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"[load] Skipped unreadable image: {image_path.name}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Use HOG for enrolment too — keeps the embedding space consistent
        locs = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, locs)

        if encs:
            name = image_path.stem.upper()
            known_encodings.append(encs[0])
            known_names.append(name)
            print(f"[load] Loaded: {name}")
        else:
            print(f"[load] No face detected in: {image_path.name}")


# ---------------------------------------------------------------------------
# Liveness — blink only
# ---------------------------------------------------------------------------

def _eye_aspect_ratio(eye_points: list) -> float | None:
    if len(eye_points) < 6:
        return None
    p = np.array(eye_points, dtype=np.float32)
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h  = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else None


def _average_ear(landmarks: dict) -> float | None:
    ears = [
        e for e in (
            _eye_aspect_ratio(landmarks.get("left_eye",  [])),
            _eye_aspect_ratio(landmarks.get("right_eye", [])),
        )
        if e is not None
    ]
    return sum(ears) / len(ears) if ears else None


def _new_liveness_state() -> dict:
    return {
        "closed_frames": 0,
        "last_seen":     0.0,
        "stage":         "waiting_close",   # waiting_close → waiting_open → verified
        "verified_until": 0.0,
    }


def update_liveness(name: str, landmarks: dict) -> tuple[bool, str]:
    """
    Blink challenge:
      1. CLOSE EYES  — hold for BLINK_CONSECUTIVE_FRAMES
      2. OPEN EYES   — eyes back open → verified
    """
    now   = time.monotonic()
    state = liveness_state.setdefault(name, _new_liveness_state())

    # Already verified and within session window
    if state["verified_until"] > now:
        state["last_seen"] = now
        return True, "LIVE"

    # Face disappeared too long → full reset
    if now - state["last_seen"] > CHALLENGE_TIMEOUT_SECONDS:
        state.clear()
        state.update(_new_liveness_state())

    state["last_seen"] = now

    ear = _average_ear(landmarks)
    if ear is None:
        return False, "FACE NOT CLEAR"

    stage = state["stage"]

    # ── Stage 1: waiting for eyes to close ──────────────────────────────
    if stage == "waiting_close":
        if ear < BLINK_EAR_THRESHOLD:
            state["closed_frames"] += 1
            if state["closed_frames"] >= BLINK_CONSECUTIVE_FRAMES:
                state["stage"] = "waiting_open"
                return False, "OPEN EYES"
        else:
            state["closed_frames"] = 0
        return False, "CLOSE EYES"

    # ── Stage 2: waiting for eyes to open again ──────────────────────────
    if stage == "waiting_open":
        if ear >= BLINK_EAR_THRESHOLD:
            state["stage"]         = "verified"
            state["verified_until"] = now + LIVE_SESSION_SECONDS
            return True, "LIVE"
        return False, "OPEN EYES"

    # Unexpected stage — reset
    state.clear()
    state.update(_new_liveness_state())
    return False, "CLOSE EYES"


# ---------------------------------------------------------------------------
# Attendance helpers
# ---------------------------------------------------------------------------

def _ensure_attendance_folder() -> None:
    ATTENDANCE_FOLDER.mkdir(parents=True, exist_ok=True)


def get_csv_path(date_str: str | None = None) -> Path:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    return ATTENDANCE_FOLDER / f"Attendance_{date_str}.csv"


def _ensure_csv(path: Path) -> None:
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Time", "Date"])
        print(f"[attendance] Created: {path.name}")


def mark_attendance(name: str) -> None:
    global _attendance_cache_file, _attendance_marked_names

    path = get_csv_path()
    _ensure_csv(path)

    with _attendance_lock:
        if _attendance_cache_file != path:
            with open(path, "r", newline="") as f:
                _attendance_marked_names = {row["Name"] for row in csv.DictReader(f)}
            _attendance_cache_file = path

        if name not in _attendance_marked_names:
            now = datetime.now()
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow([name, now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d")])
            _attendance_marked_names.add(name)
            print(f"[attendance] Marked: {name}")


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _get_center_crop(frame: np.ndarray, ratio: float = CENTER_CROP_RATIO):
    h, w = frame.shape[:2]
    cw, ch = int(w * ratio), int(h * ratio)
    l, t = (w - cw) // 2, (h - ch) // 2
    return frame[t: t + ch, l: l + cw], (l, t, l + cw, t + ch)


# ---------------------------------------------------------------------------
# Frame analysis  (HOG detector — no TF, no RetinaFace)
# ---------------------------------------------------------------------------

def analyze_frame(frame: np.ndarray) -> list:
    crop, (crop_left, crop_top, _, _) = _get_center_crop(frame)

    small_bgr = cv2.resize(crop, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

    # ── HOG detection — fast on CPU ──────────────────────────────────────
    face_locations = face_recognition.face_locations(small_rgb, model="hog",
                                                      number_of_times_to_upsample=HOG_UPSAMPLE)
    if not face_locations:
        return []

    # ── Identify verified faces first to skip unnecessary encoding ───────
    #    (encoding is the most expensive per-face step)
    now = time.monotonic()
    results = []

    face_encodings  = face_recognition.face_encodings(small_rgb, face_locations)
    face_landmarks  = face_recognition.face_landmarks(small_rgb, face_locations)

    for idx, (enc, loc) in enumerate(zip(face_encodings, face_locations)):
        name    = "UNKNOWN"
        status  = "UNKNOWN"
        is_live = False

        if known_encodings:
            matches   = face_recognition.compare_faces(known_encodings, enc,
                                                       tolerance=FACE_MATCH_TOLERANCE)
            distances = face_recognition.face_distance(known_encodings, enc)
            best      = int(np.argmin(distances))

            if matches[best]:
                name = known_names[best]

                # Fast-path: still within verified session — skip EAR compute
                state = liveness_state.get(name)
                if state and state.get("verified_until", 0) > now:
                    is_live, status = True, "LIVE"
                    mark_attendance(name)
                else:
                    lm = face_landmarks[idx] if idx < len(face_landmarks) else {}
                    is_live, status = update_liveness(name, lm)
                    if is_live:
                        mark_attendance(name)

        top, right, bottom, left = [int(v / DETECTION_SCALE) for v in loc]
        top    += crop_top;  bottom += crop_top
        left   += crop_left; right  += crop_left

        if name == "UNKNOWN":
            label, color = "UNKNOWN", COLOR_UNKNOWN
        elif is_live:
            label, color = f"{name} - LIVE", COLOR_LIVE
        else:
            label, color = f"{name} - {status}", COLOR_PENDING

        results.append({"box": (top, right, bottom, left), "color": color, "label": label})

    return results


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_results(frame: np.ndarray, results: list) -> None:
    _, (cl, ct, cr, cb) = _get_center_crop(frame)
    cv2.rectangle(frame, (cl, ct), (cr, cb), DRAW_CROP_BOX_COLOR, 1)

    for r in results:
        top, right, bottom, left = r["box"]
        color, label = r["color"], r["label"]
        cv2.rectangle(frame, (left, top), (right, bottom), color, DRAW_BOX_THICKNESS)
        cv2.rectangle(frame, (left, bottom - DRAW_LABEL_BAR_HEIGHT), (right, bottom),
                      color, cv2.FILLED)
        cv2.putText(frame, label,
                    (left + DRAW_LABEL_OFFSET_X, bottom - DRAW_LABEL_OFFSET_Y),
                    DRAW_FONT, DRAW_FONT_SCALE, (255, 255, 255), DRAW_FONT_THICKNESS)


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

def _camera_reader() -> None:
    global _latest_frame
    while True:
        for _ in range(CAMERA_GRAB_SKIP):
            _camera.grab()
        ok, frame = _camera.read()
        if ok:
            with _frame_lock:
                _latest_frame = (frame, time.monotonic())
        else:
            time.sleep(0.1)


def _analysis_worker() -> None:
    global _latest_results
    while True:
        with _frame_lock:
            snapshot = _latest_frame

        if snapshot is not None:
            frame, ts = snapshot
            if time.monotonic() - ts <= FRAME_STALE_SECONDS:
                try:
                    results = analyze_frame(frame.copy())
                except Exception:
                    traceback.print_exc()
                    results = []
                with _results_lock:
                    _latest_results = results

        time.sleep(ANALYSIS_INTERVAL_SECONDS)


def generate_frames():
    while True:
        with _frame_lock:
            snapshot = _latest_frame

        if snapshot is None or time.monotonic() - snapshot[1] > FRAME_STALE_SECONDS:
            time.sleep(STREAM_SLEEP_SECONDS)
            continue

        frame = snapshot[0].copy()
        with _results_lock:
            results = list(_latest_results)

        draw_results(frame, results)
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(STREAM_SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/attendance_data")
def attendance_data():
    path = get_csv_path()
    _ensure_csv(path)
    with _attendance_lock:
        with open(path, "r", newline="") as f:
            records = list(csv.DictReader(f))
    return jsonify(records)


@app.route("/all_dates")
def all_dates():
    _ensure_attendance_folder()
    dates = sorted(
        [
            f.stem.replace("Attendance_", "")
            for f in ATTENDANCE_FOLDER.iterdir()
            if f.name.startswith("Attendance_") and f.suffix == ".csv"
        ],
        reverse=True,
    )
    return jsonify(dates)


@app.route("/attendance_data/<date>")
def attendance_by_date(date: str):
    path = get_csv_path(date)
    if not path.exists():
        return jsonify([])
    with _attendance_lock:
        with open(path, "r", newline="") as f:
            records = list(csv.DictReader(f))
    return jsonify(records)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _release_camera() -> None:
    if _camera is not None and _camera.isOpened():
        _camera.release()
        print("[camera] Released.")


def startup() -> None:
    global _camera

    _ensure_attendance_folder()
    load_known_faces()

    _camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not _camera.isOpened():
        _camera = cv2.VideoCapture(0)
    if not _camera.isOpened():
        raise RuntimeError("Could not open camera device 0.")

    _camera.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    _camera.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    _camera.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    atexit.register(_release_camera)

    threading.Thread(target=_camera_reader,   daemon=True, name="camera-reader").start()
    threading.Thread(target=_analysis_worker, daemon=True, name="analysis-worker").start()
    print("[startup] Ready.")


if __name__ == "__main__":
    startup()
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False)