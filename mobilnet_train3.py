#%% Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torch.optim.lr_scheduler import StepLR

writer = SummaryWriter()

#%% Data Loaders
def get_data_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = torchvision.datasets.ImageFolder(
        root="/Users/velibilir/Desktop/Staj/PyTorch/kodlar/catvsdog/newimages/train_set",
        transform=transform)
    
    test_set = torchvision.datasets.ImageFolder(
        root="/Users/velibilir/Desktop/Staj/PyTorch/kodlar/catvsdog/newimages/test_set",
        transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

#%% Model
num_classes = 4
model = models.mobilenet_v2(pretrained=True)

# Fine-tuning: Sadece son 2 bloğu eğitime aç
for name, param in model.features.named_parameters():
    if "18" in name or "17" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Yeni sınıflandırıcı
model.classifier = nn.Sequential(
    nn.Dropout(0.35),
    nn.Linear(model.last_channel, num_classes)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#%% Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

#%% Test Fonksiyonu
def test_model(model, data_loader, dataset_type="test", writer=None, epoch=0):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"{dataset_type} accuracy: {accuracy:.2f} %")
    
    if writer:
        writer.add_scalar(f'Accuracy/{dataset_type}', accuracy, epoch)
    
    return accuracy

#%% Eğitim Fonksiyonu
def train_model(model, train_loader, test_loader, criterion, optimizer, writer=None, epochs=10):
    print("Training başladı...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
        if writer:
            writer.add_scalar("Loss/train", avg_loss, epoch)

        # Test doğruluğu
        test_accuracy = test_model(model, test_loader, "Test", writer=None, epoch=epoch)
        
        # Eğitim doğruluğu
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total

        if writer:
            writer.add_scalars("Accuracy", {
                "Train": train_accuracy,
                "Test": test_accuracy
            }, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        scheduler.step()

#%% Ana Program
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    epochs = 20

    train_model(model, train_loader, test_loader, criterion, optimizer, writer=writer, epochs=epochs)
    
    print("\n--- Final Test Accuracy ---")
    test_model(model, test_loader, "Final Test", writer=writer, epoch=epochs)

    print("--- Final Train Accuracy ---")
    test_model(model, train_loader, "Final Train", writer=writer, epoch=epochs)


    torch.save(model.state_dict(), "mobilenetv2_weights4.pth")

    writer.close()
