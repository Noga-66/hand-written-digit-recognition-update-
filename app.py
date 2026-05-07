import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Digit Detector",
    page_icon="✏️",
    layout="centered"
)

st.markdown("""
<style>
    .main { direction: ltr; }
    .big-digit {
        font-size: 80px;
        font-weight: bold;
        text-align: center;
        color: #6c63ff;
        line-height: 1;
    }
    .conf-text {
        text-align: center;
        color: #888;
        font-size: 14px;
    }
    .stProgress > div > div { background-color: #6c63ff; }
</style>
""", unsafe_allow_html=True)

st.title("✏️ Handwritten Digit Detector")
st.caption("Draw any digit from 0 to 9 and see the prediction instantly")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")

try:
    model = load_model()
    model_ok = True
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info("Make sure `mnist_model.keras` file is in the same folder as `app.py`")
    model_ok = False

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("#### 🖊️ Draw here")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("#### 🔍 Result")

    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data.astype(np.uint8)
        has_drawing = img_array[..., :3].sum() > 0

        if has_drawing and model_ok:
            img = Image.fromarray(img_array).convert("L")
            img = img.resize((28, 28), Image.LANCZOS)
            img_np = np.array(img).astype("float32") / 255.0
            img_np = img_np.reshape(1, 28, 28, 1)

            probs = model.predict(img_np, verbose=0)[0]
            top = int(np.argmax(probs))
            confidence = float(probs[top]) * 100

            st.markdown(f'<div class="big-digit">{top}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="conf-text">Model confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Probability for each digit:**")
            for i, p in enumerate(probs):
                label = f"**{i}**" if i == top else str(i)
                st.progress(float(p), text=f"{label} — {p*100:.1f}%")
        else:
            st.markdown('<div class="big-digit" style="color:#333;">?</div>', unsafe_allow_html=True)
            st.markdown('<div class="conf-text">Draw a digit on the left</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
#### About this app
This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits. The model was built using TensorFlow and Keras, and it achieves over 99% accuracy on the test set.
            """)