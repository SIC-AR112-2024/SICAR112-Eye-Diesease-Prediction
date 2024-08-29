import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets  # Import both transforms and datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import urllib
import glob
from random import shuffle
import seaborn as sns
from sklearn.metrics import confusion_matrix

def hello_world():
    print('hello, I am an imported python file')

def gen_cm(model, dataloader, paths):
    plt.figure(figsize=(8, 6))

    # Move model to CPU
    device = torch.device("cpu")
    model.to(device)

    # Make predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print unique values in all_labels to check the actual labels
    print(set(all_labels))

    # Generate confusion matrix using the correct labels (replace with your actual labels)
    actual_labels = list(set(all_labels))  # Get unique labels from your data
    cm = confusion_matrix(all_labels, all_preds, labels=actual_labels)
    print(actual_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=paths, yticklabels=paths)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
