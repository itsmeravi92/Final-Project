import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Weapon Detection System", layout="wide")
st.title(" Weapon Detection Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

source_radio = st.sidebar.radio(
    "Select Input Source",
    ["Live Camera", "Upload File (Image/Video)"]
)

# ---------------- MODEL SELECTION ----------------
st.sidebar.header("Model Selection")

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_options = {
    "Best Model": os.path.join(BASE_DIR, "models", "best.onnx"),
    "Best Wave": os.path.join(BASE_DIR, "models", "best-wave.onnx"),
    "Yolo Model": os.path.join(BASE_DIR, "models", "yolov8n.pt")
}

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    list(model_options.keys())
)

model_path = model_options[selected_model_name]

st.sidebar.write(f"Using Model: {selected_model_name}")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_path)

# ---------------- PROCESS FUNCTION ----------------
def process_frame(frame, conf):
    results = model.predict(frame, conf=conf)
    annotated_frame = results[0].plot()
    return annotated_frame

# ---------------- LIVE CAMERA ----------------
if source_radio == "Live Camera":
    st.header("Webcam Feed")

    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av

    class WeaponDetector(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed = process_frame(img, confidence)
            return av.VideoFrame.from_ndarray(processed, format="bgr24")

    webrtc_streamer(
        key="weapon-detection",
        video_processor_factory=WeaponDetector,
        media_stream_constraints={"video": True, "audio": False},
    )

# ---------------- FILE UPLOAD ----------------
else:
    st.header("Upload Image or Video")

    uploaded_file = st.file_uploader(
        "Choose a file...",
        type=["jpg", "jpeg", "png", "mp4", "avi"]
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]

        # ---------- IMAGE ----------
        if file_type == 'image':
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            st.subheader("Detection Result")

            processed_img = process_frame(img_array, confidence)
            st.image(processed_img, use_column_width=True)

        # ---------- VIDEO ----------
        elif file_type == 'video':
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = process_frame(frame, confidence)

                st_frame.image(processed)

            cap.release()
