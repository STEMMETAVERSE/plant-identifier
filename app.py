import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import os

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

st.title("🌿 Plant Identifier (Stable Fix)")

uploaded_file = st.file_uploader("Upload plant image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("Identify Plant"):

        # ✅ FINAL FIX: USE RAW BYTES DIRECTLY
        image_bytes = uploaded_file.getvalue()

        with st.spinner("Analyzing plant..."):

            response = client.image_classification(
                model="google/vit-base-patch16-224",
                image=image_bytes
            )

        st.subheader("Prediction")
        st.write(response)
