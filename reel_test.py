#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 15:25:22 2025

@author: velibilir
"""

# %% import libraries
import torch
import cv2
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision  
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import mobilenet_v2

writer = SummaryWriter()

# %% veri yükleyici
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    reel_set = torchvision.datasets.ImageFolder(
        root="/Users/velibilir/Desktop/Staj/PyTorch/kodlar/catvsdog/newimages/reel", 
        transform=transform)

    reel_loader = torch.utils.data.DataLoader(reel_set, batch_size=batch_size, shuffle=False)
    return reel_loader

# %% MobileNet modeli
model = mobilenet_v2(weights=None, num_classes=4)
model.load_state_dict(torch.load("mobilenetv2_weights4.pth", map_location="cpu"))
model.eval()

# %% test fonksiyonu
def test_model(model, test_loader, dataset_type="Test", writer=None, epochs=5):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for epoch in range(epochs):
            for images, labels in test_loader:
                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if writer:
                    writer.add_scalar('Accuracy/test', 100 * correct / total, epoch)

        print(f"{dataset_type} accuracy: {100 * correct / total:.2f} %")

# %% main
if __name__ == "__main__":
    reel_loader = get_data_loaders()
    test_model(model, reel_loader, dataset_type="Test", writer=writer, epochs=5)
    writer.close()
