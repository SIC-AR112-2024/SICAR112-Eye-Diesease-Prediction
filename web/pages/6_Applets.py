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


#Apparently required classes
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

class ReSNeT18(nn.Module):
    def __init__(self, num_classes):
        super(ReSNeT18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class ReSNeT34(nn.Module):
    def __init__(self, num_classes):
        super(ReSNeT34, self).__init__()
        self.model = models.resnet34(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class ReSNeT50(nn.Module):
    def __init__(self, num_classes):
        super(ReSNeT50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(CustomResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class ResidualBlock(nn.Module):

    expansion = 1  # Set to 1 for basic ResNet blocks

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


# Load the pre-trained ResNet model
def load_model1(MODEL_PATH):
    model = models.resnet50()
    model = torch.load(MODEL_PATH, map_location='cpu')
    model.eval()
    return model

# Load the pre-trained ResNet model
def load_model2(MODEL_PATH):
    model = models.resnet34()
    model = torch.load(MODEL_PATH, map_location='cpu')
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
    
    # Resize the image
    st.sidebar.header("Resize Options")
    width = st.sidebar.slider("Width", min_value=100, max_value=1000, value=300)
    height = st.sidebar.slider("Height", min_value=100, max_value=1000, value=300)
    
    # Resize the image
    resized_image = image.resize((width, height))
    
    # Display the image in the app
    st.image(resized_image, caption="Retinal Fundus Image", use_column_width=True)
    
    # Optionally, you can process the image here
    st.write("Image successfully uploaded and displayed! Pending Analysis...")
    
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    if st.button('Click here for prediction'):
        with torch.no_grad():
            outputs = [model(img_tensor) for model in modelz]

        labels = load_labels("labels/labels.txt")

        predicted_indices = [torch.argmax(output, dim=1).item() for output in outputs]
        predicted_labels = [labels[idx] for idx in predicted_indices]

        st.write(f"Predicted Label for the resnet50 model: {predicted_labels[0]}")
        st.write(f"Predicted Label for the resnet34 model: {predicted_labels[1]}")
        st.write(f"Predicted Label for the ResNet-AR112 Ensemble model: {predicted_labels[2]}")