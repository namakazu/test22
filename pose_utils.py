from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

CONNECTIONS = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]


def ensure_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return MODEL_PATH

    try:
        with urllib.request.urlopen(MODEL_URL, timeout=40) as response:
            MODEL_PATH.write_bytes(response.read())
    except Exception as exc:
        raise RuntimeError(
            "姿勢推定モデルを取得できませんでした。通信状態を確認して再実行してください。"
        ) from exc
    return MODEL_PATH


def create_landmarker():
    model_path = ensure_model()
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def detect_landmarks(frame_bgr: np.ndarray, landmarker) -> Optional[List]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(image)
    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]


def _point(lms, idx: int, w: int, h: int) -> Tuple[float, float]:
    return (lms[idx].x * w, lms[idx].y * h)


def _angle(a, b, c) -> float:
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cosv = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosv))


def _line_angle(a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    deg = abs(math.degrees(math.atan2(dy, dx)))
    return min(deg, 180 - deg)


def calculate_metrics(lms, frame_shape) -> Dict[str, float]:
    h, w = frame_shape[:2]
    p = lambda i: _point(lms, i, w, h)

    ls, rs = p(L_SHOULDER), p(R_SHOULDER)
    lh, rh = p(L_HIP), p(R_HIP)
    mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    torso_angle_h = _line_angle(mid_hip, mid_shoulder)
    forward_lean_from_vertical = abs(90 - torso_angle_h)

    nose = p(NOSE)
    shoulder_width = abs(rs[0] - ls[0]) / max(w, 1)

    return {
        "肩ライン傾斜(°)": round(_line_angle(ls, rs), 1),
        "腰ライン傾斜(°)": round(_line_angle(lh, rh), 1),
        "上体の傾き・鉛直基準(°)": round(forward_lean_from_vertical, 1),
        "左肘角度(°)": round(_angle(p(L_SHOULDER), p(L_ELBOW), p(L_WRIST)), 1),
        "右肘角度(°)": round(_angle(p(R_SHOULDER), p(R_ELBOW), p(R_WRIST)), 1),
        "左膝角度(°)": round(_angle(p(L_HIP), p(L_KNEE), p(L_ANKLE)), 1),
        "右膝角度(°)": round(_angle(p(R_HIP), p(R_KNEE), p(R_ANKLE)), 1),
        "肩幅(画面比)": round(shoulder_width, 4),
        "頭中心X": round(nose[0] / max(w, 1), 4),
        "頭中心Y": round(nose[1] / max(h, 1), 4),
        "腰中心X": round(mid_hip[0] / max(w, 1), 4),
        "腰中心Y": round(mid_hip[1] / max(h, 1), 4),
    }


def draw_pose(frame_bgr: np.ndarray, lms) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    def valid(idx: int) -> bool:
        lm = lms[idx]
        vis = getattr(lm, "visibility", 1.0)
        return vis is None or vis >= 0.35

    pts = {}
    for i, lm in enumerate(lms):
        if valid(i):
            pts[i] = (int(lm.x * w), int(lm.y * h))

    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(out, pts[a], pts[b], (255, 255, 255), 3, cv2.LINE_AA)

    for idx in [
        L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
        L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    ]:
        if idx in pts:
            cv2.circle(out, pts[idx], 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(out, pts[idx], 4, (255, 255, 255), -1, cv2.LINE_AA)

    return out
