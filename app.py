import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
import os

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

st.title("🌿 Plant Identifier (Fixed Stable Version)")

uploaded_file = st.file_uploader("Upload plant image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    if st.button("Identify Plant"):

        # 🔥 IMPORTANT FIX: Convert PIL → bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        image_bytes = img_byte_arr.getvalue()

        with st.spinner("Analyzing plant..."):

            response = client.image_classification(
                model="google/vit-base-patch16-224",
                image=image_bytes
            )

        st.subheader("Prediction")
        st.write(response)
