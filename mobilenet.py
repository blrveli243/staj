#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:52:25 2025

@author: velibilir
"""
#%%import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
writer = SummaryWriter()
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    train_set = torchvision.datasets.ImageFolder(
        root="/Users/velibilir/Desktop/staj/PyTorch/kodlar/mabilenet/cat_dog"
        ,transform=transform)
    
    test_set = torchvision.datasets.ImageFolder(
        root="/Users/velibilir/Desktop/staj/PyTorch/kodlar/mabilenet/cat_dog",
    transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle =True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
    
    return train_loader,test_loader
#%% visualize datase
def imshow(img):
    img = img /2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0)))
    plt.axis("off")
    plt.show()
    
def get_sample_images(train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels

def visualize(n):
    train_loader, test_loader = get_data_loaders()
    images, labels = get_sample_images(train_loader)
    
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))  # Tek satır, n kolon
    
    for i in range(n):
        img = images[i] / 2 + 0.5
        np_img = img.numpy().transpose((1, 2, 0))
        axes[i].imshow(np_img)
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off")

    plt.tight_layout() # Bu olmadan başlıklar taşabilir
    plt.show()
visualize(5)
#%% CNN mobilenet model

# Önceden eğitilmiş MobilNetV2 modelini yükle
model = models.mobilenet_v2(pretrained=True)

# Sınıf sayını belirt (örneğin: 11 sınıf için)
num_classes = 2

# Son katmanı (classifier) yeniden tanımla
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

define_loss_and_optimizer = lambda model :(
    nn.CrossEntropyLoss(), # multi class classification problem
    optim.Adam(model.parameters(), lr=0.001) #öğrenme sırasında momentum (bir önceki gradyanlardan alınan ortalama) ve adaptif öğrenme oranı (her parametreye özel hız) kullanır.
    )

#%% training 
def train_model(model ,train_loader, criterion, optimizer,writer = None,epochs = 5, ):
    model.train()
    train_losses = []
    print("Training başladı..")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            
            loss= criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"epoch:{epoch+1}/{epochs}, Loss: {avg_loss:.5f}") 
        writer.add_scalar("Loss/train", loss.item(), epoch)

        
        
        writer.add_scalar('Loss/train', avg_loss, epoch)

    

        for param in model.features.parameters():
            param.requires_grad = False



#%%% testing
def test_model(model,test_loader,dataset_type,writer=None):
    
    model.eval()#değerlendirme modu
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"{dataset_type} accuracy: {100 * correct / total} %")
      
        writer.add_scalar('Accuracy/test', 100 * correct / total, 0)
        writer.add_scalar("Accuracy/train", correct / total)

        
#%%main program
if __name__=="__main__":
    
    
    train_loader,test_loader = get_data_loaders()
    
    visualize(5)
  
    
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs=3,writer=writer)


    test_model(model, test_loader, dataset_type = "test",writer=writer)
    writer.close()
