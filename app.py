import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import os
import tempfile

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

st.title("🌿 Plant Identifier")

uploaded_file = st.file_uploader("Upload plant image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Identify Plant"):

        with st.spinner("Analyzing plant..."):

            # write temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # PASS FILE PATH
            response = client.image_classification(
                model="google/vit-base-patch16-224",
                image=tmp_path
            )

        st.subheader("Prediction")
        st.write(response)
