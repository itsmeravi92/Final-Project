import os
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile


st.set_page_config(
    page_title="WeaponSense AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# iddhar css ka kaam hai
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0a0c10; color: #e2e8f0; }

    [data-testid="stSidebar"] {
        background: #0f1117 !important;
        border-right: 1px solid #1e2533;
    }

    .main-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: #f8fafc;
        text-transform: uppercase;
        padding: 0.5rem 0 0.2rem 0;
        border-bottom: 2px solid #ef4444;
        margin-bottom: 1.5rem;
    }

    .sub-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        background: #141820;
        border: 1px solid #1e2533;
        border-left: 3px solid #ef4444;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    .metric-label {
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .badge-safe {
        display: inline-block;
        background: #052e16;
        color: #4ade80;
        border: 1px solid #166534;
        border-radius: 4px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .badge-threat {
        display: inline-block;
        background: #450a0a;
        color: #f87171;
        border: 1px solid #991b1b;
        border-radius: 4px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .detection-item {
        background: #141820;
        border: 1px solid #1e2533;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }

    .info-box {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0.5rem 0 1rem 0;
    }

    .stButton > button {
        background: #ef4444 !important;
        color: white !important;
        border: none !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        border-radius: 4px !important;
    }

    .stButton > button:hover { background: #dc2626 !important; }

    hr { border-color: #1e2533 !important; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# yaha mene code change kr kiya h mr pradumn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_OPTIONS = {
    "Best Model (ONNX)":  os.path.join(BASE_DIR, "models", "best.onnx"),
    "Best Wave (ONNX)":   os.path.join(BASE_DIR, "models", "best-wave.onnx"),
    "YOLOv8n (PyTorch)":  os.path.join(BASE_DIR, "models", "yolov8n.pt"),
}

# praumn don ye sidebar h 
with st.sidebar:
    st.markdown('<div class="sub-header">⚙ Configuration</div>', unsafe_allow_html=True)

    selected_model_name = st.selectbox(
        "Detection Model",
        list(MODEL_OPTIONS.keys()),
        help="Choose the AI model for weapon detection"
    )
    model_path = MODEL_OPTIONS[selected_model_name]

    st.markdown("---")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Detections below this score will be ignored"
    )

    st.markdown("---")

    source = st.radio(
        "Input Source",
        ["📷  Webcam", "📁  Upload File"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f'<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value" style="font-size:0.85rem;">{selected_model_name}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{int(confidence*100)}%</div></div>', unsafe_allow_html=True)


@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"❌ Could not load model at `{model_path}`. Make sure the `models/` folder exists next to `app.py`.")
    st.stop()


def run_detection(frame, conf):
    results = model.predict(frame, conf=conf, verbose=False)
    annotated = results[0].plot()
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        score = float(box.conf[0])
        detections.append({"label": label, "confidence": score})
    return annotated, detections


def render_detections(detections):
    if not detections:
        st.markdown('<span class="badge-safe">✔ No Threats Detected</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-threat">⚠ {len(detections)} Threat(s) Detected</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for d in detections:
            st.markdown(f"""
            <div class="detection-item">
                <span style="color:#f8fafc; font-weight:600;">{d['label'].upper()}</span>
                &nbsp;&nbsp;
                <span style="color:#ef4444; font-family:'Rajdhani',sans-serif; font-weight:700;">{d['confidence']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)


st.markdown('<div class="main-header">🔍 WeaponSense AI — Detection Dashboard</div>', unsafe_allow_html=True)

# ye webcam mode h 
if "Webcam" in source:
    st.markdown('<div class="sub-header">📷 Webcam Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Point your camera at the subject and click <strong>Take Photo</strong>. The model will analyze the captured frame for weapons.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        picture = st.camera_input("", label_visibility="collapsed")

    with col2:
        st.markdown('<div class="sub-header">Results</div>', unsafe_allow_html=True)
        result_slot = st.empty()

        if picture is not None:
            image = Image.open(picture)
            img_array = np.array(image)

            with st.spinner("Analyzing..."):
                processed, detections = run_detection(img_array, confidence)

            col1.image(processed, use_container_width=True, caption="Annotated Output")

            with result_slot.container():
                render_detections(detections)
                if detections:
                    st.markdown("<br>", unsafe_allow_html=True)
                    avg = sum(d["confidence"] for d in detections) / len(detections)
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Detections</div><div class="metric-value">{len(detections)}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Confidence</div><div class="metric-value">{avg*100:.1f}%</div></div>', unsafe_allow_html=True)
        else:
            result_slot.markdown('<div style="color:#475569; font-size:0.85rem; margin-top:1rem;">Awaiting capture...</div>', unsafe_allow_html=True)

# yaha se file upload krdena
else:
    st.markdown('<div class="sub-header">📁 File Upload Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Upload an image or video. Supported: JPG, PNG, MP4, AVI.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=["jpg", "jpeg", "png", "mp4", "avi"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]

        # ── IMAGE ──────────────────────────────
        if file_type == 'image':
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns([1.4, 1])

            with col1:
                st.image(image, caption="Original", use_container_width=True)

            with st.spinner("Running detection..."):
                processed, detections = run_detection(img_array, confidence)

            with col1:
                st.image(processed, caption="Detection Output", use_container_width=True)

            with col2:
                st.markdown('<div class="sub-header">Results</div>', unsafe_allow_html=True)
                render_detections(detections)
                if detections:
                    st.markdown("<br>", unsafe_allow_html=True)
                    avg = sum(d["confidence"] for d in detections) / len(detections)
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Total Detections</div><div class="metric-value">{len(detections)}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Confidence</div><div class="metric-value">{avg*100:.1f}%</div></div>', unsafe_allow_html=True)
# yaha se video krna upload
        elif file_type == 'video':
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.flush()

            col1, col2 = st.columns([1.4, 1])

            with col1:
                st.markdown('<div class="sub-header">Video Feed</div>', unsafe_allow_html=True)
                frame_slot = st.empty()

            with col2:
                st.markdown('<div class="sub-header">Live Stats</div>', unsafe_allow_html=True)
                stats_slot = st.empty()
                progress = st.progress(0)
                stop = st.button("⏹  Stop Processing")

            cap = cv2.VideoCapture(tfile.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frame_n = 0
            all_det = []

            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed, detections = run_detection(frame_rgb, confidence)
                all_det.extend(detections)

                frame_slot.image(processed, use_container_width=True)
                frame_n += 1
                progress.progress(min(frame_n / total, 1.0))

                with stats_slot.container():
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Frame</div><div class="metric-value">{frame_n} / {total}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Total Detections</div><div class="metric-value">{len(all_det)}</div></div>', unsafe_allow_html=True)
                    if detections:
                        st.markdown('<span class="badge-threat">⚠ Weapon in Frame</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-safe">✔ Clear</span>', unsafe_allow_html=True)

            cap.release()

            if frame_n > 0:
                st.success(f"✅ Done — {frame_n} frames analyzed, {len(all_det)} total detections.")
