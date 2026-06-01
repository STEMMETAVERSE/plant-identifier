import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import os

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

st.title("🌿 Plant Identifier (ViT Model)")

uploaded_file = st.file_uploader("Upload plant image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Identify Plant"):

        with st.spinner("Analyzing plant..."):

            response = client.image_classification(
                model="google/vit-base-patch16-224",
                image=uploaded_file.getvalue()
            )

        st.subheader("🌱 Prediction")
        st.write(response)
