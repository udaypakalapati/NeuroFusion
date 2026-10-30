
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="NeuroFusion", layout="wide")

st.title("🧠 NeuroFusion")
st.subheader("Multi-Modal MRI Fusion and Deep Learning for Brain Tumor Detection and Segmentation")

st.info("Research prototype only. Not intended for clinical diagnosis.")

# Model architecture
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.enc1 = ConvBlock(1, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        self.output = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)


# Load model
from pathlib import Path

@st.cache_resource
def load_model():
    model = UNet(num_classes=4)

    model_path = Path(__file__).parent / "neurofusion_unet.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model


try:
    model = load_model()
    st.success("U-Net model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# MRI upload
uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((240, 240))

    img_array = np.array(image).astype(np.float32)
    img_array = (img_array - img_array.min()) / (
        img_array.max() - img_array.min() + 1e-8
    )

    input_tensor = torch.tensor(
        img_array, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_mask = torch.argmax(prediction, dim=1).squeeze().numpy()

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_array, caption="Original MRI", clamp=True)

    with col2:
        st.image(
            predicted_mask.astype(np.uint8) * 60,
            caption="Predicted Segmentation",
            clamp=True
        )

    st.subheader("Pixel Distribution")

    total_pixels = predicted_mask.size

    for i in range(4):
        percentage = np.sum(predicted_mask == i) / total_pixels * 100
        st.write(f"Class {i}: {percentage:.2f}%")

    st.warning(
        "These are dataset class labels and pixel percentages, "
        "not verified tumor grades or diagnostic probabilities."
    )
