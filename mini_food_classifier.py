"""
11 farklı yemek kategorisini içeren bir görsel veri setini kullanarak, bu görsellerin sınıflarını tahmin eden bir 
CNN modeli geliştirmek.
"""
#%% import libraris

import torch
import torch.nn as nn #sinir ağı katmanları için
import torch.optim as optim #optimizasyon algoritması
import torchvision # görünti işleme için
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

  #load dataset
def get_data_loaders(batch_size = 64): # her iterasyonda işlenecek veri sayısı
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(), # görüntüyü tensöre çevirir
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))# rgb kanalları normalize et
        ])  

   # Food101 veri setini indir ve eğititm ve test veri setini oluştur.
    train_set = torchvision.datasets.ImageFolder(
    root='/Users/velibilir/Desktop/staj/PyTorch/kodlar/mini_food_classifier/food11/train',
    transform=transform)

    test_set = torchvision.datasets.ImageFolder(
    root='/Users/velibilir/Desktop/staj/PyTorch/kodlar/mini_food_classifier/food11/test',
    transform=transform)

   # dataloader (veri yüklemek)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader   = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#%% visualize dataset

def imshow(img):
    # verileri normalize etmeden önce geri dönüştür
    img = img / 2 + 0.5  # normalize'in tersi
    np_img = img.numpy()  # tensörden numpy'a çevir
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  #BGR -> RGB
    plt.axis("off")
    plt.show()

def get_sample_images(train_loader):  # veri kümesinden örnek görseller almak için fonksiyon
    data_iter = iter(train_loader)
    images, labels = next(data_iter)    
    return images, labels

def visualize(n):
    train_loader, test_loader_ = get_data_loaders()
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

# Görselleştir
#visualize(5)

 
#%% Build CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,padding=1)

        self.relu = nn.ReLU() #aktivasyon fonk oluşturur
        
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)#2*2 ye pooling katmanı
        
        self.conv2= nn.Conv2d(32, 64, kernel_size =3,padding=1)#64 filtre ikinci layer
        self.dropout = nn.Dropout(0.2)
        self.fc1 =nn.Linear(64*56*56, 128)
        self.fc2 = nn.Linear(128,11)# çıktı katmanı 
        
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x))) #ilk convulation blok
        x = self.pool(self.relu(self.conv2(x))) # ikinci covulation blok

        x = x.view(x.size(0), -1) #Bu satır PyTorch'ta tensörü düzleştirmek (flatten etmek) için kullanılır ve CNN → Fully Connected geçişinde.
    
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
   
define_loss_and_optimizer = lambda model :(
    nn.CrossEntropyLoss(), # multi class classification problem
    optim.Adam(model.parameters(), lr=0.001) #öğrenme sırasında momentum (bir önceki gradyanlardan alınan ortalama) ve adaptif öğrenme oranı (her parametreye özel hız) kullanır.
    )
"""
SDG:her adımda basitçe gradyanı hesaplar ve ağırlıkları buna göre günceller.

"""

#%% Training
def train_model(model, train_loader, criterion, optimizer, epochs = 5):
    model.train()
    train_losses = []
    print("Training başladı...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for images, labels in train_loader: # tüm eğitim veri setini taramak için
            
            optimizer.zero_grad()
            outputs = model(images)
                
            loss = criterion(outputs,labels)#loss hesaplar
            loss.backward() #gradyan hesaplar geri yayılım yapar
            optimizer.step() #ogrenme = parametre yeni agırlık güncelleme

            total_loss += loss.item()
            
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"epoch:{epoch+1}/{epochs}, Loss: {avg_loss:.5f}")

     #kayip (loss) grafiği
    plt.figure()
    plt.plot(range(1, epochs+1),train_losses, marker="o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

#%% Testing
def test_model(model,test_loader,dataset_type):
    
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




#%% Main program
if __name__=="__main__":
    
    
    train_loader,test_loader = get_data_loaders()
    
    visualize(5)
    
    model=CNN()
    
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs=1)


    test_model(model, test_loader, dataset_type = "test")
    # test_model(model, train_loader, dataset_type= "training")