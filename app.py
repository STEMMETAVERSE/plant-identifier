import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
import os

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

st.title("🌿 Plant Identifier (Stable HF Router Fix)")

uploaded_file = st.file_uploader("Upload plant image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    if st.button("Identify Plant"):

        # 🔥 CRITICAL FIX: proper file-like object WITH NAME
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        img_bytes.name = "image.jpg"   # 👈 THIS FIXES YOUR ERROR

        with st.spinner("Analyzing plant..."):

            response = client.image_classification(
                model="google/vit-base-patch16-224",
                image=img_bytes
            )

        st.subheader("Prediction")
        st.write(response)
