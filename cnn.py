"""
Problemin tanimmi: CIFAR10 ile siniflandirma problemi
CNN
"""

#%% impport libraris
import torch
import torch.nn as nn# sinir agı katmanları için
import torch.optim as optim #optimizasyon algoritması içim
import torchvision #gorüntü isleme
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# load dataset
# optional: cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loaders(batch_size = 64): # her iterasyonda islenecek veri sayısi
    transform = transforms.Compose([
        transforms.ToTensor(), # görüntüyü tensöre çevirir
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))# rgb kanalları normalize et
        
        ])
    
    # CİFAR10 veri setini indir ve eğitim ve test veri setini olustur.
    train_set = torchvision.datasets.CIFAR10(root="./data",train=True, download=True,transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data",train= False, download=True, transform=transform) 

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_set , batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set , batch_size = batch_size, shuffle = False)

    return train_loader,test_loader    

#%% visualize dataset

def imshow(img):
    # verileri normalize etmeden önce geri dönüştür
    img = img / 2 + 0.5  # normalize'in tersi
    np_img = img.numpy()  # tensörden numpy'a çevir
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  
    plt.axis("off")
    plt.show()
 
def get_sample_images(train_loader):  # veri kümesinden örnek görseller almak için fonksiyon
    data_iter = iter(train_loader)#batchini alıyo
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

# Görselleştir
#visualize(5)

 


#%% build CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
              
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU() #aktvasyon fonksiyonu Lineer olmayanlık kazandırır, yani modelin karmaşık desenleri öğrenmesini sağlar.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)#2*2 ye pooling katmanı                       
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 filtre ikinci layer
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64*8*8,128)
        self.fc2 = nn.Linear(128,10) # output layer
        
        # image 3*32*32 -> conv(32) -> relu(32) ->pool(16)
        # conv(16) -> relu(16) -> pool(8) -> image(8*8)
        
        
    def  forward(self,x):
        
        """
         image 3*32*32 -> conv(32) -> relu(32) ->pool(16)
         conv(16) -> relu(16) -> pool(8) -> image(8*8)
         flatten
         fc1 -> relu -> dropout
         fc2 -> output
        """
        x = self.pool(self.relu(self.conv1(x)))#ilk convulation blok
        x = self.pool(self.relu(self.conv2(x)))#ikinci convulation blok
        # x = x.view(-1, 64*8*8)
        x = x.view(x.size(0), -1)# düzleştiri linear hale getirir.flattende aynı işi yapr

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)#output
        return x


#model = CNN()

# define loss function and optimizer
define_loss_and_optimizer = lambda model :(
    
    nn.CrossEntropyLoss(), # multi class classification problem
    optim.SGD(model.parameters(), lr = 0.001, momentum= 0.9) # sdg
    
   )




# %% training
def train_model(model,train_loader, criterion, optimizer, epochs = 5 ):
    model.train()#eğitim modunu açtık.
    train_losses = [] # loss değerlerini saklamak için 
    
    for epoch in range(epochs):
        total_loss = 0 # toplam loss'u tutmak için
        
        for images, labels in train_loader: # tum eğitim veri setini taramak için
            
            optimizer.zero_grad()
            outputs = model(images)# Forward prediction output = etiket, label, class
            
            loss = criterion(outputs,labels) #loss değeri hesapla
            loss.backward() # geri yayilim (gradyan hesapklama)
            optimizer.step() #ogrenme = parametre yeni agırlık güncelleme
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)# ortalama kayip hesaplama
        train_losses.append(avg_loss)
        print(f"epoch:{epoch+1}/{epochs}, Loss: {avg_loss:.5f}")
        
    
    #kayip (loss) grafiği
    plt.figure()
    plt.plot(range(1, epochs+1),train_losses, marker="o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


#model = CNN()
#criterion, optimizer = define_loss_and_optimizer(model)
#train_model(model, train_loader, criterion, optimizer, epochs=10)



#%% testing

def test_model(model,test_loader, dataset_type):
    
    model.eval() # değerlindirme modu
    correct = 0 #doğru tahmin sayacı
    total = 0 # toplam veri sayaci
    
    with torch.no_grad(): # gradyan hesaplamsını kapat,
        for images, labels in test_loader: # test veri satini kullanarak degerlendirme
        
            outputs = model(images) # prediction
        
            _, predicted = torch.max(outputs, 1) #en yuksek olasılıklı sınıfı seç
            total += labels.size(0) # toplam veri sayisi
            correct += (predicted == labels).sum().item() # doğru tahminleri say
            
        print(f"{dataset_type} accuracy: {100 * correct / total} %")
                
#test_model(model, test_loader, dataset_type = "test")
#test_model(model, test_loader, dataset_type= "training")
            
            
# %% Main program
 
if __name__ == "__main__":
    
    
    train_loader, test_loader = get_data_loaders()

    visualize(2)
    
    model = CNN()
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs=20)

    test_model(model, test_loader, dataset_type = "test")
    test_model(model, train_loader, dataset_type= "training")
    

    

   





























