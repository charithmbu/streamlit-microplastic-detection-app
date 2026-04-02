import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "Microplastic_Yolov8_Model.pt"
EXAMPLE_DIR = "Example_images"
PIXEL_TO_NM = 100
RISK_THRESHOLD = 15

# ---------------- LOAD MODEL (CACHED ⚡) ----------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

with st.spinner("🔄 Loading YOLOv8 model..."):
    model = load_model()

# ---------------- UI ----------------
st.set_page_config(page_title="Microplastic Detection System", layout="wide")
st.title("🧪 Microplastic Detection System (YOLOv8)")

st.markdown("### 📥 Choose Input Method")

input_mode = st.radio(
    "Select Input Type:",
    ["Upload Image", "Use Example Image", "Capture from Camera"]
)

img_bytes = None

# ---------------- CAMERA INPUT ----------------
if input_mode == "Capture from Camera":
    cam = st.camera_input("Capture image from microscope / camera")
    if cam:
        img_bytes = cam.getvalue()
        st.image(Image.open(cam), caption="Captured Image")

# ---------------- EXAMPLE IMAGE ----------------
elif input_mode == "Use Example Image":
    if os.path.exists(EXAMPLE_DIR):
        images = os.listdir(EXAMPLE_DIR)
        if images:
            img_name = st.selectbox("Select example image", images)
            path = os.path.join(EXAMPLE_DIR, img_name)

            with open(path, "rb") as f:
                img_bytes = f.read()

            st.image(Image.open(path), caption="Example Image")
        else:
            st.warning("No images found in Example_images folder")
    else:
        st.warning("Example_images folder not found")

# ---------------- UPLOAD IMAGE ----------------
else:
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if file:
        img_bytes = file.read()
        st.image(Image.open(file), caption="Uploaded Image")

# ---------------- LOCAL DETECTION ----------------
if img_bytes:
    st.subheader("🚀 Running Detection...")

    # Convert bytes to OpenCV image
    file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ Error reading image")
        st.stop()

    # Run YOLO
    results = model(img)[0]

    boxes = []
    annotated_img = img.copy()

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        w = x2 - x1
        h = y2 - y1

        boxes.append({
            "width": w,
            "height": h
        })

        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    total_count = len(boxes)

    # ---------------- RISK LOGIC ----------------
    risk_score = total_count
    status = "⚠️ Risky" if risk_score > RISK_THRESHOLD else "✅ Safe"

    # Convert BGR → RGB
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    st.image(annotated_img, caption="Detected Microplastics", width=600)

    # -------- SUMMARY --------
    st.subheader("📊 Detection Summary")
    st.write(f"Total Microplastics Detected: **{total_count}**")
    st.write(f"Risk Score: **{risk_score}**")
    st.write(f"Final Status: **{status}**")

    # -------- SIZE LOGIC (UNCHANGED) --------
    sizes_nm = []
    for box in boxes:
        w_nm = box["width"] * PIXEL_TO_NM
        h_nm = box["height"] * PIXEL_TO_NM
        sizes_nm.append(np.sqrt(w_nm * h_nm))

    if sizes_nm:
        min_size = min(sizes_nm)
        max_size = max(sizes_nm)
        avg_size = sum(sizes_nm) / len(sizes_nm)

        min_thresh = min_size * 1.10
        max_thresh = max_size * 0.90

        min_count = sum(s <= min_thresh for s in sizes_nm)
        max_count = sum(s >= max_thresh for s in sizes_nm)
        avg_count = total_count - min_count - max_count

        st.subheader("📦 Size Category Counts")
        st.write(f"Min Size: {min_count}")
        st.write(f"Average Size: {avg_count}")
        st.write(f"Max Size: {max_count}")

        fig, ax = plt.subplots()
        ax.bar(["Min", "Avg", "Max"], [min_count, avg_count, max_count])
        ax.set_ylabel("Count")
        st.pyplot(fig)