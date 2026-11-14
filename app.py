"""Member 3 - Streamlit frontend for DermAssist."""
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import analyze

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "dermassist_model.keras")

st.set_page_config(page_title="DermAssist", page_icon="🩺", layout="wide")
st.title("🩺 DermAssist")
st.caption("Skin lesion screening: image processing + deep learning. "
           "Educational prototype - NOT a medical diagnosis.")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH)


model = load_model()
threshold = st.sidebar.slider("Malignant decision threshold", 0.1, 0.9, 0.5, 0.05)
st.sidebar.info("Lower threshold = more sensitive (catches more malignant cases, "
                "more false alarms).")
if model is None:
    st.sidebar.warning("No trained model found. Run train.py first. "
                       "Image-processing tabs still work.")

file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])
if file is None:
    st.info("Upload an image to begin.")
    st.stop()

img_bgr = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Could not read this image.")
    st.stop()

res = analyze(img_bgr)
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

tab1, tab2, tab3 = st.tabs(["1. Preprocessing & Segmentation", "2. ABCD Analysis",
                            "3. AI Classification + Grad-CAM"])

with tab1:
    c = st.columns(4)
    c[0].image(rgb(res["original"]), caption="Original", width="stretch")
    c[1].image(rgb(res["clean"]), caption="Hair removed", width="stretch")
    c[2].image(rgb(res["enhanced"]), caption="CLAHE enhanced", width="stretch")
    c[3].image(rgb(res["overlay"]), caption="Otsu lesion boundary", width="stretch")
    st.image(res["mask"], caption="Segmentation mask", width=300)

with tab2:
    f = res["features"]
    if f is None:
        st.warning("No lesion could be segmented in this image.")
    else:
        m = st.columns(4)
        m[0].metric("Asymmetry", f["asymmetry"], help="0 = symmetric")
        m[1].metric("Border (compactness)", f["compactness"], help="1.0 = perfect circle")
        m[2].metric("Distinct colours", f["n_colors"])
        m[3].metric("Diameter (px)", f["diameter_px"])
        st.subheader(f"ABCD score: {f['abcd_score']} / 4 warning flags")
        st.table(pd.DataFrame({"Criterion": list(f["flags"].keys()),
                               "Flagged": ["⚠️ Yes" if v else "✅ No" for v in f["flags"].values()]}))
        st.caption("Thresholds are simple heuristics for demonstration.")
        with st.expander("All raw feature values"):
            st.json({k: v for k, v in f.items() if k != "flags"})

with tab3:
    if model is None:
        st.warning("Train the model first (train.py).")
    else:
        from gradcam import gradcam_overlay
        img_rgb = rgb(res["original"])
        overlay, p_mal = gradcam_overlay(model, img_rgb)
        is_mal = p_mal >= threshold
        conf = p_mal if is_mal else 1 - p_mal
        if is_mal:
            st.error(f"⚠️ Higher-risk lesion (malignant) - confidence {conf:.1%}")
        else:
            st.success(f"✅ Likely benign - confidence {conf:.1%}")
        st.progress(float(p_mal), text=f"P(malignant) = {p_mal:.3f}")
        c = st.columns(2)
        c[0].image(img_rgb, caption="Input", width="stretch")
        c[1].image(overlay, caption="Grad-CAM: regions driving the prediction", width="stretch")

st.divider()
st.caption("Always consult a qualified dermatologist for any skin concern.")
