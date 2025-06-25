#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN : tekrarlayan sinir ağları zaman serilerinde kullanıyoruz

veri seti seçme

"""
# %% veri seti olusturma and görsellestirme

import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def generate_data(seq_length = 50, num_samples = 1000):

    """
    example: 3lu paket
    sequance: [2,3,4] giris dizilerini saklamajk için
    targets: [5] hedef değerleri saklamk için
    
    """
    x = np.linspace(0, 100, num_samples)#0-100 arası num_samples kadar veri olustur
    y = np.sin(x)
    sequance = [] # giris dizilerini saklamak için
    targets = [] # hedef değerleri saklamak için

    for i in range(len(x) - seq_length):
        sequance.append(y[i:i+seq_length])# input
        targets.append(y[i+seq_length])# input değerden sonra gelen deger
    
    plt.figure(figsize=(8,4))
    plt.plot(x,y,label = 'sin(t)',color='b',linewidth=2)
    plt.title('Sinüs Dalga Grafigi')
    plt.xlabel('Zaman(radyan)')
    plt.ylabel("Genlik")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return np.array(sequance),np.array(targets)

sequance, targets = generate_data()

# %% RNN modelini olustur

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers =1):
        super(RNN, self).__init__()
        #input size: giriş boyutu
        #hidden_size: rnn nin gizli katman cell sayısı
        #num layers: layer sayısı
        #output_size: çıktı boyutu
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)#RNN layer
    
        self.fc = nn.Linear(hidden_size, output_size) # fully connected layer: output
    
    def forward(self, x):
        out, _=self.rnn(x)#rnn girdiyi ver çıktıyı al
        out = self.fc(out[:,-1,:]) #son zaman adımındaki çıktıyı al ve fc layera bağla
        return out
    
model = RNN(1,16,1,1)
        
        
        
# %% rnn training

#hyperparameters

seq_length = 50 # input dizisinin boyutu
input_size = 1 #input dizisinin boyutu
hidden_size = 16 #rnn nin gizli katmandaki düğüğm sayısı
output_size = 1 # tahmin edilen değerin çıktının boyutu
num_layers = 1 # rnn katman sayısi
epochs = 20 #modelin kaç kez tüm veri üzerinde eğitilecegi
batch_size = 32 #her bireğitim adımında kaçorneğin kulanalıcagı
learning_rate = 0.001 # optimizasyon algoritmasında öğrenme hızı ve ya oranı

#veriyi hazırla
x, y = generate_data(seq_length)
x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)#pytorch tensörüne cevir ve boyut ekle
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)#pytorch tensörüne cevir ve boyut ekle

dataset = torch.utils.data.TensorDataset(x,y)#pytorch dataset olusturma
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)# veri yükleyici

#modeli tanımla
model = RNN(input_size, hidden_size, output_size,num_layers)
criterion = nn.MSELoss()# ortalsms kare hatası
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for batch_x, batch_y in dataLoader:
        optimizer.zero_grad()
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)# model tahmini ile gerçek tahminn karsılastırır ve loss hesaplar
        loss.backward()
        optimizer.step()#agırlıkları güncelle
    print(f"Epoch:{epoch+1}/{epochs},loss: {loss.item():.4f}")
        

#%% rnn test and evulation

# test için veri olusturma
x_test = np.linspace(100, 110,seq_length).reshape(1, -1) # ilk test verisi
y_test = np.sin(x_test) # test verimizin gerçek değeri

x_test2 = np.linspace(120, 130,seq_length).reshape(1, -1) # ikinci test verisi
y_test2 = np.sin(x_test2) 
        
   # from numpy to tensor 
x_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(-1)
x_test2 = torch.tensor(y_test2,dtype=torch.float32).unsqueeze(-1)

# modeli kullanarak prediction yap
model.eval()
prediction1 = model(x_test).detach().numpy()# ilk test verisi için tahmin
prediction2 = model(x_test2).detach().numpy()

#sonucları görsellestir
plt.figure()
plt.plot(np.linspace(0,100,len(y)),y,marker = "o",label = "training dataset")
plt.plot(x_test.numpy().flatten(),marker = "o", label = "Test 1")
plt.plot(x_test2.numpy().flatten(),marker = "o", label = "Test 2")

        
plt.plot([110], prediction1.flatten(), "ro", label="Prediction 1")
plt.plot([110], prediction2.flatten(), "go", label="Prediction 2")

plt.legend()
plt.show()     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        