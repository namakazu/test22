from __future__ import annotations

import hashlib
import hmac
import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from rule_coach import diagnose_rules
from pose_utils import calculate_metrics, create_landmarker, detect_landmarks, draw_pose
from video_utils import encode_jpeg, frame_at, get_video_info, prepare_video


st.set_page_config(
    page_title="Golf Form Coach",
    page_icon="🏌️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 760px;
        padding-top: .8rem;
        padding-bottom: 5rem;
        padding-left: .9rem;
        padding-right: .9rem;
    }
    h1 { font-size: 1.65rem !important; margin-bottom: .25rem !important; }
    h2 { font-size: 1.28rem !important; margin-top: 1.3rem !important; }
    h3 { line-height: 1.25 !important; }
    div.stButton > button {
        min-height: 3.15rem;
        font-size: 1.02rem;
        font-weight: 700;
        border-radius: 14px;
    }
    div[data-testid="stFileUploader"] { border-radius: 14px; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    @media (max-width: 640px) {
        .block-container { padding-top: .55rem; padding-left: .7rem; padding-right: .7rem; }
        h1 { font-size: 1.48rem !important; }
        p, label { font-size: .97rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return default


def _password_gate() -> None:
    expected = _secret("APP_PASSWORD")
    if not expected:
        return

    if st.session_state.get("authenticated"):
        return

    st.title("🏌️ Golf Form Coach")
    st.caption("プライベートテスト")
    entered = st.text_input("パスワード", type="password")
    if st.button("開く", type="primary", use_container_width=True):
        if hmac.compare_digest(entered, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います。")
    st.stop()


_password_gate()

st.title("🏌️ Golf Form Coach")
st.caption("初心者向けフォーム改善 — API不要・1回に直すのは最大2点")

with st.expander("📹 撮り方", expanded=False):
    st.markdown(
        """
- スマホを**固定**して、頭から足先まで入れる
- できれば60fpsで撮る
- 1スイングだけ、前後に2〜3秒余白を入れる
- 正面：胸の正面付近から
- 後方：ボールと目標方向のライン後方から

最初は**正面動画1本**で十分です。
        """
    )

st.subheader("1. スイング動画を選ぶ")
st.caption("iPhone/Androidの標準カメラで撮影して、ここから動画を選びます。")
uploaded = st.file_uploader(
    "動画を選択",
    type=["mp4", "mov", "m4v", "avi"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

if not uploaded:
    st.info("10秒前後のスイング動画を1本選んでください。")
    st.stop()

video_bytes = uploaded.getvalue()
if len(video_bytes) > 250 * 1024 * 1024:
    st.error("動画が大きすぎます。250MB以下にしてください。")
    st.stop()

file_hash = hashlib.sha256(video_bytes).hexdigest()[:20]
suffix = Path(uploaded.name).suffix.lower() or ".mp4"
work_dir = Path(tempfile.gettempdir()) / "golf_form_coach"
work_dir.mkdir(parents=True, exist_ok=True)
raw_path = work_dir / f"{file_hash}{suffix}"
if not raw_path.exists():
    raw_path.write_bytes(video_bytes)

try:
    with st.spinner("動画を準備しています…"):
        video_path, transcoded = prepare_video(raw_path, work_dir / f"{file_hash}_normalized.mp4")
        fps, frame_count, duration = get_video_info(video_path)
except Exception as e:
    st.error(f"動画を読み込めませんでした: {e}")
    st.caption("iPhoneの場合は短い動画で再撮影して試してください。")
    st.stop()

if duration <= 0 or frame_count <= 0:
    st.error("動画情報を取得できませんでした。別の動画で試してください。")
    st.stop()
if duration > 45:
    st.warning("45秒より長い動画です。解析を軽くするため、1スイング10秒前後がおすすめです。")

st.video(str(video_path))
status = "スマホ動画を互換形式へ変換済み" if transcoded else "動画形式OK"
st.caption(f"{status} / {duration:.1f}秒 / 約{fps:.0f}fps")

with st.expander("⚙️ 撮影条件", expanded=True):
    view = st.radio("撮影方向", ["正面（Face-on）", "後方（Down-the-line）"])
    handedness = st.radio("利き打ち", ["右打ち", "左打ち"], horizontal=True)

st.subheader("2. 4つの瞬間を合わせる")
st.caption("ざっくりでOKです。動画を見ながら近い位置に合わせます。")

defaults = {
    "アドレス": duration * 0.15,
    "トップ": duration * 0.38,
    "インパクト": duration * 0.58,
    "フィニッシュ": duration * 0.82,
}

times: dict[str, float] = {}
for idx, (label, default) in enumerate(defaults.items(), start=1):
    with st.expander(f"{idx}/4　{label}", expanded=(idx == 1)):
        max_value = max(duration, 0.1)
        step = max(0.01, 1.0 / max(fps, 1.0))
        times[label] = st.slider(
            f"{label}の時刻",
            min_value=0.0,
            max_value=max_value,
            value=min(default, duration),
            step=step,
            format="%.2f 秒",
            key=f"phase_{file_hash}_{label}",
        )
        try:
            preview = frame_at(video_path, times[label])
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
        except Exception as e:
            st.warning(str(e))

st.subheader("3. フォームを解析")
if st.button("🏌️ フォーム解析を開始", type="primary", use_container_width=True):
    landmarker = None
    try:
        with st.spinner("身体の動きを確認しています…"):
            landmarker = create_landmarker()
            original_jpegs = []
            all_metrics = {}
            results = []

            for label, sec in times.items():
                frame = frame_at(video_path, sec)
                lms = detect_landmarks(frame, landmarker)
                original_jpegs.append((label, encode_jpeg(frame)))
                if lms is None:
                    results.append((label, frame, None, None))
                    all_metrics[label] = {"解析": "身体を検出できませんでした"}
                    continue
                overlay = draw_pose(frame, lms)
                metrics = calculate_metrics(lms, frame.shape)
                results.append((label, frame, overlay, metrics))
                all_metrics[label] = metrics

        st.success("骨格解析が完了しました。")
        for label, frame, overlay, metrics in results:
            with st.expander(f"✅ {label}", expanded=False):
                show = overlay if overlay is not None else frame
                st.image(cv2.cvtColor(show, cv2.COLOR_BGR2RGB), use_container_width=True)
                if metrics:
                    for key, value in metrics.items():
                        st.caption(f"{key}: {value}")
                else:
                    st.warning("このフレームでは身体を検出できませんでした。")

        st.session_state["golf_frames"] = original_jpegs
        st.session_state["golf_metrics"] = all_metrics
        st.session_state["golf_view"] = view
        st.session_state["golf_handedness"] = handedness
        st.session_state.pop("diagnosis", None)
    except Exception as e:
        st.error(f"解析に失敗しました: {e}")
        st.caption("初回だけMediaPipeのモデル取得で失敗する場合があります。再実行してください。")
    finally:
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass

if "golf_metrics" in st.session_state:
    st.subheader("4. 改善ポイント")
    st.caption("外部AI/APIは使いません。骨格データを端末・サーバー内のルールで判定します。")

    if st.button("🎯 次の10球で直すポイントを見る", type="primary", use_container_width=True):
        try:
            diagnosis = diagnose_rules(
                st.session_state["golf_metrics"],
                st.session_state.get("golf_view", view),
                st.session_state.get("golf_handedness", handedness),
            )
            st.session_state["diagnosis"] = diagnosis
        except Exception as e:
            st.error(f"診断に失敗しました: {e}")

if st.session_state.get("diagnosis"):
    st.markdown(st.session_state["diagnosis"])
    st.divider()
    st.markdown("### 次の10球")
    st.write("最優先の1点だけ意識して10球打ち、もう一度撮影してください。")

st.divider()
st.caption("MVP: 2D動画からの参考解析です。外部AI/APIは使用しません。クラブ軌道や3D角度を精密測定する機器ではありません。")
