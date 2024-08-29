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


def run_model(model, dataloaders):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
    y = []
    y_hat = []


    EPOCHS = 20
    # Check if CUDA is available after installation

    if torch.cuda.is_available():
        print("CUDA is available!")
    else:
        print("CUDA is not available. Check your installation.")
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, inp in enumerate(dataloaders['train']):
            #print(inp)
            inputs, labels = inp
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i%5 == 0 and i > 0:
                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        avg_loss = sum(losses)/len(losses)
        scheduler.step(avg_loss)
        corrects = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                corrects += (preds == labels).sum().item()

        accuracy = corrects / total
        print(f'Validation Accuracy: {accuracy:.4f}')

    # Testing loop
    model.eval()
    corrects = 0
    total = 0


    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            y.append(labels)
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            y_hat.append(outputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (preds == labels).sum().item()


    accuracy = corrects / total
    print(f'Test Accuracy: {accuracy:.4f}')





    print('Training Done')