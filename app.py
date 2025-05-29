import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# -----------------------------
# Frontend: Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Garbage Classification",
    page_icon="🗑️",
    layout="centered",
    initial_sidebar_state="auto"
)

# -----------------------------
# Title and Description
# -----------------------------
st.markdown("""
    <div style='text-align: center;'>
        <h1>🧠 Smart Garbage Classifier</h1>
        <p>Upload an image to classify it .</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# Backend: Class Labels
# -----------------------------
CLASSES = ['battery', 'biological','brown-glass','cardboard','cclothes','green-glass','metal','paper','plastic','shoes','trash','white-glass']

# -----------------------------
# Backend: Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("waste_classified.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Backend: Preprocessing
# -----------------------------
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# -----------------------------
# File Uploader
# -----------------------------
uploaded_file = st.file_uploader("📷 Upload Garbage Image", type=["jpg", "jpeg", "png"])

# -----------------------------
# Prediction Logic
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)
        predicted_class = CLASSES[prediction.item()]

    # Display Result
    st.success(f"✅ Predicted Class: **{predicted_class}**")

# -----------------------------
# Sidebar: About
# -----------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This app uses a **ResNet18** model trained to classify garbage images into:
                    'battery',
                    'biological',
                    'brown-glass',
                    'cardboard',
                    'cclothes',
                    'green-glass',
                    'metal',
                    'paper',
                    'plastic',
                    'shoes',
                    'trash',
                    'white-glass'
- 
Developed using:
- Streamlit                  
- PyTorch
- torchvision
""")
