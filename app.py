import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Import our existing backend functions
from src.noise import add_gaussian_noise, add_salt_and_pepper_noise
from src.filters import apply_mean_filter, apply_median_filter, apply_gaussian_filter, apply_nlm_filter
from src.metrics import calculate_mse, calculate_psnr

st.set_page_config(page_title="FDIP Noise Removal", layout="wide", page_icon="📸")

st.title("📸 Image Noise Removal App")
st.markdown("**FDIP Mini Project** | Test how mathematical filters remove digital noise from images.")

# --- Sidebar Controls ---
st.sidebar.header("1. Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

# Alternatively, let them pick from default data folder if no upload
default_images = [f for f in os.listdir("data") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not uploaded_file and default_images:
    selected_default = st.sidebar.selectbox("Or select a benchmark image:", default_images)
    image_path = os.path.join("data", selected_default)
    original_image = cv2.imread(image_path)
elif uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
else:
    original_image = None

if original_image is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("2. Add Noise")
    noise_type = st.sidebar.radio("Select Noise Type:", ["None", "Gaussian", "Salt & Pepper"])
    noise_intensity = st.sidebar.slider("Noise Intensity / Amount:", 0.01, 0.20, 0.05, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.header("3. Apply Filter")
    filter_type = st.sidebar.radio("Select Filter:", ["None", "Mean", "Median", "Gaussian", "NLM (Advanced)"])
    kernel_size = st.sidebar.slider("Kernel Size / Strength:", 3, 15, 5, step=2)

    # --- Processing ---
    # 1. Apply Noise
    if noise_type == "Gaussian":
        noisy_image = add_gaussian_noise(original_image, var=noise_intensity)
    elif noise_type == "Salt & Pepper":
        noisy_image = add_salt_and_pepper_noise(original_image, amount=noise_intensity)
    else:
        noisy_image = original_image.copy()

    # 2. Apply Filter
    if filter_type == "Mean":
        filtered_image = apply_mean_filter(noisy_image, kernel_size=kernel_size)
    elif filter_type == "Median":
        filtered_image = apply_median_filter(noisy_image, kernel_size=kernel_size)
    elif filter_type == "Gaussian":
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        filtered_image = apply_gaussian_filter(noisy_image, kernel_size=kernel_size, sigma=sigma)
    elif filter_type == "NLM (Advanced)":
        filtered_image = apply_nlm_filter(noisy_image, h=kernel_size)
    else:
        filtered_image = noisy_image.copy()

    # 3. Calculate Metrics
    mse = calculate_mse(original_image, filtered_image)
    psnr = calculate_psnr(original_image, filtered_image)

    # --- Layout & Display ---
    col1, col2, col3 = st.columns(3)
    
    def convert_to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img

    with col1:
        st.subheader("Original Image")
        st.image(convert_to_rgb(original_image), use_container_width=True)

    with col2:
        st.subheader(f"Noisy Image ({noise_type})")
        st.image(convert_to_rgb(noisy_image), use_container_width=True)

    with col3:
        st.subheader(f"Filtered Image ({filter_type})")
        st.image(convert_to_rgb(filtered_image), use_container_width=True)
        
        # Display Metrics Beautifully
        if mse == 0:
            st.success("✨ **Perfect Match!** (Images are identical)")
            st.metric(label="PSNR (dB)", value="∞")
            st.metric(label="MSE", value="0.00")
        else:
            if psnr > 30:
                st.success("🟢 Excellent Restoration")
            elif psnr > 22:
                st.warning("🟡 Fair Restoration")
            else:
                st.error("🔴 Poor Restoration")
                
            m_col1, m_col2 = st.columns(2)
            m_col1.metric(label="PSNR (dB)", value=f"{psnr:.2f}")
            m_col2.metric(label="MSE", value=f"{mse:.2f}")
else:
    st.info("👈 Please select or upload an image from the sidebar to begin!")
