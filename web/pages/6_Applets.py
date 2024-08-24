import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import requests
import os

st.title("Applets")
st.write("Trialing photo upload feature.")

# Load custom labels from a file
def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

paths = ["models/resnet50.pth", "models/resnet34.pth", "models/ResNet-AR112.pth"]

# Load the pre-trained ResNet model
def load_model1(MODEL_PATH):
    model = models.resnet50()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

# Load the pre-trained ResNet model
def load_model2(MODEL_PATH):
    model = models.resnet34()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def load_model3(MODEL_PATH):
    model = torch.load(MODEL_PATH, map_location='cpu')
    model.eval()
    return model

load_model = [load_model1, load_model2, load_model3]
# Define image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

modelz = []

for i in range(len(paths)):
    mod = load_model[i](paths[i])
    modelz.append(mod)



# Define image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Display the image in the app
    st.image(image, caption="Retinal Fundus Image", use_column_width=True)
    
    # Optionally, you can process the image here
    st.write("Image successfully uploaded and displayed! Pending Analysis...")
    
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    if st.button('Click here for prediction'):
        with torch.no_grad():
            outputs = [modelz[0](img_tensor), modelz[1](img_tensor), modelz[2](img_tensor)]

        labels = load_labels('labels\labels.txt')

        _, predicted_idx = [torch.max(outputs[0], 1), torch.max(outputs[1], 1), torch.max(outputs[2], 1)]
        predicted_labels = [labels[predicted_idx[0].item()], labels[predicted_idx[1].item()], labels[predicted_idx[2].item()]]

        st.write(f"Predicted Label for the resnet50 model: {predicted_labels[0]}")
        st.write(f"Predicted Label for the resnet34 model: {predicted_labels[1]}")
        st.write(f"Predicted Label for the ResNet-AR112 Ensemble model: {predicted_labels[2]}")