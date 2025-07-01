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
import torchvision.models  as models
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import GoogLeNet_Weights
#writer = SummaryWriter()


transform = transforms.Compose([
    
transforms.Resize((224, 224)),     
transforms.ToTensor(),
transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

    ])

def get_data_loader(batch_size = 64):
    
    train_set = torchvision.datasets.ImageFolder(
        root="/Users/velibilir/Desktop/Staj/PyTorch/kodlar/catvsdog/cat_dog",
        transform = transform
        )
    
    test_set = torchvision.datasets.ImageFolder(
        root="/Users/velibilir/Desktop/Staj/PyTorch/kodlar/catvsdog/cat_dog",
        transform= transform
        )
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    
    test_loader =  torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader
# %% CNN model
num_classes = 2
weights = GoogLeNet_Weights.DEFAULT  # en iyi ön tanımlı ağırlıklar
model = models.googlenet(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# model = models.mobilenet_v2( pretrained = True)
# model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

define_loss_and_optimizer = lambda model : (
    nn.CrossEntropyLoss(),
    optim.Adam(model.parameters(), lr=0.001))

# %% Training

def train_model (model, train_loader, criterion,optimizer, writer= None, epochs = 5 ):
    model.train()
    train_losses = []
    print("Training started ..")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss/len(train_loader)
        train_losses.append(avg_loss)
            
        print(f"epoch:{epoch+1}/{epochs}, Loss: {avg_loss:.5f}") 
       # writer.add_scalar("Loss/train", loss.item(), epoch)
        #writer.add_scalar('Loss/train', avg_loss, epoch)

        
# %% Testing

def test_model(model,test_loader, dataset_type, writer = None):
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"{dataset_type} accuracy: {100 * correct / total} %")
      
        #writer.add_scalar('Accuracy/test', 100 * correct / total,0)
        #writer.add_scalar("Accuracy/train", correct / total)
        
 # %%  Kamera ile test
def run_camera(model, class_names):
    cap = cv2.VideoCapture(0)

    model.eval()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        input_tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = class_names[predicted.item()]

        cv2.putText(frame, f"Tahmin: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Canli Kamera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    

# %% main program

if __name__ == "__main__":
    
    train_loader, test_loader =get_data_loader()
    
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs=1)


   # test_model(model, test_loader, dataset_type = "test",writer=writer)
    
    class_names = ['kedi', 'kopek']
    run_camera(model, class_names)

    
    #writer.close()
    

























