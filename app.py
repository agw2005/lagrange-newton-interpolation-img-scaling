# app.py

import streamlit as st
import numpy as np
import cv2
from utils import upscale_image

st.set_page_config(page_title="Image Upscaler", layout="centered")

st.title("🖼️ Image Upscaler (Lagrange/Newton)")

uploaded_file = st.file_uploader("Upload an image (can only hold up to 300x300 due to memory limitation)", type=["png", "jpg", "jpeg"])

method = st.selectbox("Choose interpolation method:", ["lagrange", "newton"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        st.error("Could not load the image. Make sure it is a valid image file.")
    else:
        # Resize input image if its height exceeds 300 pixels, preserving aspect ratio
        max_height = 300
        orig_height, orig_width = image.shape
        if orig_height > max_height:
            scale = max_height / orig_height
            new_width = int(orig_width * scale)
            image = cv2.resize(image, (new_width, max_height), interpolation=cv2.INTER_AREA)

        st.image(image, caption=f"Original Image ({image.shape[0]}x{image.shape[1]})", use_container_width =True, clamp=True)

        with st.spinner("Upscaling..."):
            upscaled = upscale_image(image, method=method)

        st.image(upscaled, caption=f"Upscaled Image ({method})({upscaled.shape[0]}x{upscaled.shape[1]})", use_container_width =True, clamp=True)

        # Download button
        result_name = f"upscaled_{method}.png"
        _, buffer = cv2.imencode(".png", upscaled)
        st.download_button(
            label="Download Upscaled Image",
            data=buffer.tobytes(),
            file_name=result_name,
            mime="image/png"
        )

