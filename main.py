from __future__ import annotations

import base64
import hashlib
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from pose_utils import calculate_metrics, create_landmarker, detect_landmarks, draw_pose
from rule_coach import diagnose_rules
from video_utils import frame_at, get_video_info, prepare_video

app = FastAPI(title="Golf Form Coach")

MAX_BYTES = 220 * 1024 * 1024
WORK_DIR = Path(tempfile.gettempdir()) / "golf_form_coach_fastapi"
WORK_DIR.mkdir(parents=True, exist_ok=True)

HTML = r"""
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover" />
<title>Golf Form Coach</title>
<style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue","Noto Sans JP",sans-serif;color:#162018;background:#f6f7f5}
body{margin:0}.wrap{max-width:720px;margin:auto;padding:18px 14px 80px}
.card{background:white;border-radius:18px;padding:16px;margin:12px 0;box-shadow:0 4px 20px #0000000a}
h1{font-size:25px;margin:4px 0 6px}.sub{color:#657066;font-size:14px;line-height:1.5}
label{display:block;font-weight:700;margin:12px 0 6px}
input[type=file],select{width:100%;box-sizing:border-box;font-size:16px;padding:12px;border:1px solid #d8ddd9;border-radius:12px;background:#fff}
button{width:100%;font-size:17px;font-weight:800;padding:15px;border:0;border-radius:14px;background:#2f6b4f;color:white;margin-top:16px}
button:disabled{opacity:.55}
#status{white-space:pre-wrap;font-size:14px;line-height:1.5;margin-top:12px}
.result{white-space:pre-wrap;line-height:1.65;background:#f8faf8;border-radius:14px;padding:14px}
.frame{background:#fff;border-radius:14px;padding:12px;margin:12px 0}
.frame img{width:100%;border-radius:12px;display:block}
.metrics{font-size:12px;color:#5d675f;white-space:pre-wrap;margin-top:8px}
.progress{display:none;height:8px;background:#e7ebe8;border-radius:9px;overflow:hidden;margin-top:12px}
.progress > div{height:100%;width:35%;background:#2f6b4f;animation:move 1.2s infinite ease-in-out}
@keyframes move{0%{transform:translateX(-120%)}100%{transform:translateX(300%)}}
.note{font-size:13px;color:#6d756f}
</style>
</head>
<body>
<div class="wrap">
  <h1>🏌️ Golf Form Coach</h1>
  <div class="sub">初心者向けフォーム改善。外部AI/APIは使いません。</div>
  <div class="card">
    <button onclick="location.href='/live'">📷 リアルタイム練習モード</button>
    <div class="note" style="margin-top:8px">スマホを置いて、1球ごとに自動判定。動画は端末内で解析します。</div>
  </div>
  <div class="sub">または、撮影済み動画をアップロードして解析できます。</div>

  <div class="card">
    <label>① スイング動画</label>
    <input id="video" type="file" accept="video/*,.mov,.mp4,.m4v" />
    <div id="fileinfo" class="note"></div>

    <label>② 撮影方向</label>
    <select id="view">
      <option>正面（Face-on）</option>
      <option>後方（Down-the-line）</option>
    </select>

    <label>③ 利き打ち</label>
    <select id="handedness">
      <option>右打ち</option>
      <option>左打ち</option>
    </select>

    <button id="go">動画を解析する</button>
    <div class="progress" id="progress"><div></div></div>
    <div id="status"></div>
  </div>

  <div id="output"></div>

  <div class="note">
    ※ 10秒前後・1スイングだけの動画がおすすめです。頭から足先まで入るようにスマホを固定してください。
  </div>
</div>

<script>
const video = document.getElementById('video');
const info = document.getElementById('fileinfo');
const go = document.getElementById('go');
const status = document.getElementById('status');
const progress = document.getElementById('progress');
const output = document.getElementById('output');

video.addEventListener('change', () => {
  const f = video.files[0];
  if (!f) { info.textContent=''; return; }
  const mb = f.size / 1024 / 1024;
  info.textContent = f.name + ' / ' + mb.toFixed(1) + ' MB';
  if (mb > 220) info.textContent += ' — 大きすぎます。220MB以下にしてください。';
});

go.addEventListener('click', async () => {
  const f = video.files[0];
  if (!f) { status.textContent='動画を選んでください。'; return; }
  if (f.size > 220*1024*1024) { status.textContent='220MBを超えています。短く撮り直してください。'; return; }

  go.disabled = true;
  progress.style.display='block';
  status.textContent='アップロード中…\n動画が大きい場合は1〜2分かかることがあります。';
  output.innerHTML='';

  const fd = new FormData();
  fd.append('file', f);
  fd.append('view', document.getElementById('view').value);
  fd.append('handedness', document.getElementById('handedness').value);

  try {
    const res = await fetch('/analyze', {method:'POST', body:fd});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '解析に失敗しました');

    status.textContent = '解析完了 / ' + data.duration.toFixed(1) + '秒';
    let html = '<div class="card"><h2>今回の診断</h2><div class="result"></div></div>';
    html += '<div class="card"><h2>解析フレーム</h2>';
    for (const fr of data.frames) {
      html += '<div class="frame"><b>' + fr.label + '</b><img src="' + fr.image + '">';
      html += '<div class="metrics">' + escapeHtml(fr.metrics_text) + '</div></div>';
    }
    html += '</div>';
    output.innerHTML = html;
    output.querySelector('.result').textContent = data.diagnosis;
    output.scrollIntoView({behavior:'smooth', block:'start'});
  } catch (e) {
    status.textContent='エラー: ' + e.message;
  } finally {
    progress.style.display='none';
    go.disabled=false;
  }
});

function escapeHtml(s){
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/live", response_class=HTMLResponse)
def live():
    return Path("live.html").read_text(encoding="utf-8")

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    view: str = Form("正面（Face-on）"),
    handedness: str = Form("右打ち"),
):
    suffix = Path(file.filename or "swing.mov").suffix.lower()
    if suffix not in {".mov", ".mp4", ".m4v", ".avi"}:
        raise HTTPException(400, "MOV / MP4 / M4V の動画を選んでください。")

    raw = WORK_DIR / f"upload_{next(tempfile._get_candidate_names())}{suffix}"
    total = 0
    try:
        with raw.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(413, "動画が220MBを超えています。短く撮り直してください。")
                out.write(chunk)

        digest = hashlib.sha256(raw.read_bytes()[:2 * 1024 * 1024]).hexdigest()[:14]
        normalized = WORK_DIR / f"{digest}_normalized.mp4"
        video_path, _ = prepare_video(raw, normalized)
        fps, frame_count, duration = get_video_info(video_path)

        if duration <= 0:
            raise HTTPException(400, "動画の長さを取得できませんでした。")
        if duration > 45:
            raise HTTPException(400, "45秒以内の動画にしてください。1スイング10秒前後がおすすめです。")

        phase_times = {
            "アドレス": duration * 0.15,
            "トップ": duration * 0.38,
            "インパクト": duration * 0.58,
            "フィニッシュ": duration * 0.82,
        }

        landmarker = create_landmarker()
        metrics_all = {}
        frames_out = []
        try:
            for label, sec in phase_times.items():
                frame = frame_at(video_path, sec)
                lms = detect_landmarks(frame, landmarker)
                if lms is None:
                    metrics = {"解析": "身体を検出できませんでした"}
                    show = frame
                else:
                    metrics = calculate_metrics(lms, frame.shape)
                    show = draw_pose(frame, lms)

                metrics_all[label] = metrics
                ok, buf = cv2.imencode(".jpg", show, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    raise RuntimeError("画像変換に失敗しました")
                image = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
                metrics_text = "\n".join(f"{k}: {v}" for k, v in metrics.items())
                frames_out.append({
                    "label": label,
                    "image": image,
                    "metrics_text": metrics_text,
                })
        finally:
            landmarker.close()

        diagnosis = diagnose_rules(metrics_all, view, handedness)
        return JSONResponse({
            "ok": True,
            "duration": duration,
            "fps": fps,
            "diagnosis": diagnosis,
            "frames": frames_out,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"解析に失敗しました: {exc}")
    finally:
        try:
            raw.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/selftest")
def selftest():
    import numpy as np
    landmarker = None
    try:
        landmarker = create_landmarker()
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        lms = detect_landmarks(frame, landmarker)
        return {
            "ok": True,
            "mediapipe_initialized": True,
            "opencv_version": cv2.__version__,
            "pose_detected_on_blank_frame": lms is not None,
        }
    finally:
        if landmarker is not None:
            landmarker.close()
