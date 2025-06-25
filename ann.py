#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:36:48 2025

@author: velibilir
"""
#problem tanimi:mnist veri seti ile rakam siniflandırma
#MNIST
#ANN:yapay sinir ağlari

#%%library
import torch # pytorch kutuphanesi , trnsör islemlereri
import torch.nn as nn # yapay sinir ağlarını tanımlamak için kullan
import torch.optim as optim # optimizayon algoritmalai içeren modül
import torchvision # görüntü isleme ve pre-defined modelleri içerir
import torchvision.transforms as transforms # görüntü donusümleri yapmak
import matplotlib.pyplot as plt #gorsellestirme

# optional: cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#veri seti yükleme , data loading
def get_data_loaders(batch_size = 64): # her iterasyonda işlenecek veri miktarı, batch size
    
   transform = transforms.Compose([
        transforms.ToTensor(), # görüntüyü tensöre çevirir 0-255 -> 0-1 olceklendiririr .
        transforms.Normalize((0.5,),(0.5,)) #piksel değerlerini -1 ile 1 arasınıa ölçekler.
        
        ])
   # mnist veri setini inidr ve test kumelerini olustur
   train_set = torchvision.datasets.MNIST(root="./data", train=True,download=True, transform= transform)
   test_set = torchvision.datasets.MNIST(root="./data", train=False,download=True, transform= transform)
   
   # pytorch veri yükleyici olustur.
   train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
   test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
   
   return train_loader,test_loader
train_loader, test_loader = get_data_loaders()


#data visualization
def visulize_samples(loader,n):
    images, labels = next(iter(loader)) # ilk batch den goruntu ve etiketleri alalim
    fig, axes = plt.subplots(1,n, figsize = (10,5)) #n farkli goruntu icin gorsellestirme
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap = "gray") # gorelei gri tonlsms olarak goster.
        axes[i].set_title(f"Label:{labels[i].item()}") # goruntuye ait sinif etşketini baslık olarak yaz.
        axes[i].axis("off")# eksenleri gizle
        
    plt.show()
    
visulize_samples(train_loader,2)

#%%define ann model


#yapay sinir ağı clsssı

class NeuralNetwork(nn.Module): #pytorch un nn.module sınıfından miras aliyor

   def __init__(self): #insa etmek için gerekli olan bilesenleri tanımla
       super(NeuralNetwork,self).__init__()
       
       self.flatten = nn.Flatten() #elinizde bulunan görüntüleri (2D) vektor haliine cevirelim (1D) ->28*28 =784
       
       self.fc1 = nn.Linear(28*28,128) # ilk tam bağlı katman olusturduk
       
       self.relu = nn.ReLU() # aktivasyon fonksiyonu olustur
       
       self.fc2 = nn.Linear(128,64) # ikinci tam bağlı katman
       
       self.fc3 = nn.Linear(64, 10) # çıktı katmanı olusur
       
   def forward(self,x): #forward propagation : ileri  yayilim , giris olarak  x = goruntu alsin
   
       x = self.flatten(x) # initial x = 28*28 görüntü  - > duzlesir 784 vektör haline gelir.
       x = self.fc1(x)
       x = self.relu(x)
       x = self.fc2(x)
       x = self.relu(x)
       x = self.fc3(x)
       
       return x # modelimizin çıktısını return edelim
   
# create model and compile
model = NeuralNetwork().to(device)

#kayip fonksiyonu ve optimizasyon algoritmasini belirle
define_loss_and_optimizer = lambda model : (
    nn.CrossEntropyLoss(),#multi class classification problemsloss function,
    optim.Adam(model.parameters(),lr = 0.001) #ubdate weights with adam
    )
#criterion, optimizer = define_loss_and_optimizer(model)

#%%train
def train_model(model,train_loader, criterion, optimizer , epochs = 10):
    model.train() # modelimizi eğitim moduna alalım.
    train_losses = [] # her bir epoch sonucunda elde edilen loss değerlerini saklamak için bir liste
    
    for epoch in range(epochs): # Belirtilen epoch seviyesi kadar eğitim yapalım.
        total_loss = 0 # toplam kayip değeri.
          
        
        for images, labels in train_loader:# Tüm eğitim verileri uzerinde itarasyon gerçekleşir.
        
            #images, labels = images.to.(device), labels.to(device)# veerileri cihaza tasi.
            
            optimizer.zero_grad() # gradyantları sıfırla
            proditions = model(images) # Modeli uygula. forward propogition.
            loss = criterion(proditions,labels) # Loss hesapla 
            loss.backward() # geri yayılım yani gradyan hesaplama
            optimizer.step() # ağırlıkları güncelleme(update weights)
    

            total_loss = total_loss + loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")

    #loss graph
    plt.Figure()
    plt.plot(range(1, epochs+1), train_losses, marker = "o", linestyle = "-", label =  "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend()
    plt.show()

#train_model(model,train_loader, criterion, optimizer, epochs = 5)






#%%test
def test_model(model,test_loader):
    model.eval()#modelimizi değerlendirme moduna al
    correct = 0 # Doğru tahmin sayaci
    total = 0 # toplam veri sayaci
    
    with torch.no_grad(): #gradyan hesaplamayı kapatıık gerek yok .
        for images, labels in test_loader: # Test veri kümesini döngüye al
            
            predictions = model(images)
            _, predicted = torch.max(predictions, 1)# en yüksekolasikli sınıfın etiketini bul
            
            total += labels.size(0)#toplam veri sayisını güncelle
            correct += (predicted == labels).sum().item() #doğru tahminleri say
            
    print(f"Test Accuracy: {100*correct/total:.3f}%")
    
#test_model(model, test_loader)

#%% Main
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()# veri yükleyiciilerini al
    visulize_samples(train_loader, 5)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model,train_loader, criterion, optimizer, epochs = 7)
    test_model(model, test_loader)
























