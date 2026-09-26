from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def prepare_video(input_path: str | Path, output_path: str | Path) -> tuple[Path, bool]:
    """Convert mobile video to browser/OpenCV-friendly H.264 MP4 when ffmpeg is available."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path, True

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return input_path, False

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-an",
        "-vf",
        "scale=min(1280\\,iw):-2:force_original_aspect_ratio=decrease",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return output_path, True
    except (subprocess.SubprocessError, OSError):
        pass

    try:
        output_path.unlink(missing_ok=True)
    except OSError:
        pass
    return input_path, False


def get_video_info(path: str | Path) -> Tuple[float, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("動画を開けませんでした")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or fps > 240:
        fps = 30.0
    duration = frames / fps if frames > 0 else 0.0
    cap.release()
    return fps, frames, duration


def frame_at(path: str | Path, sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("動画を開けませんでした")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"{sec:.2f}秒地点のフレームを取得できませんでした")
    return frame


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEGエンコードに失敗しました")
    return buf.tobytes()
