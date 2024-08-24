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

class EnsembleModel(nn.Module):
    def __init__(self, models, class_weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

        # Ensure that class_weights is a list of tensors, one for each model
        if class_weights is not None:
            assert len(class_weights) == len(models), "Number of weight sets must match number of models"
            self.class_weights = [torch.tensor(w, dtype=torch.float32) for w in class_weights]
        else:
            # If no class-specific weights are provided, use equal weights (i.e., no change)
            num_classes = models[0].fc.out_features  # Assuming all models have the same number of classes
            self.class_weights = [torch.ones(num_classes, dtype=torch.float32) for _ in models]

    def forward(self, x):
        weighted_outputs = []
        for i, model in enumerate(self.models):
            output = model(x)
            weight = self.class_weights[i].to(output.device)  # Ensure weights are on the correct device
            weighted_output = output * weight  # Apply class-specific weights to the model's output
            weighted_outputs.append(weighted_output)

        # Stack the weighted outputs and average them across the models
        stacked_outputs = torch.stack(weighted_outputs)
        final_output = torch.mean(stacked_outputs, dim=0)
        return final_output

# Example usage:
# Assuming you have 4 classes and 3 models
# Example class-specific weights for each model:
class_weights_model1 = [0.9241706161,	0.8933649289,	0.8672985782,	0.8317535545]
class_weights_model2 = [0.9573459716,	0.9928909953,	0.9194312796,	0.9265402844]
class_weights_model5 = [0.8744075829,	0.86492891,	0.691943128,	0.7772511848]


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
    model = EnsembleModel()
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