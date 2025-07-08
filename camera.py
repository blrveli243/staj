
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
from torchvision.models import mobilenet_v2
import cv2
from PIL import Image
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = mobilenet_v2(weights=None, num_classes=4)

model.load_state_dict(torch.load("mobilenetv2_weights4.pth", map_location="cpu"))

model.eval()



def run_camera(model, class_names, threshold=0.8):
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
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            label_idx = predicted.item()
            label = class_names[label_idx]
            confidence_val = confidence.item()

            if confidence_val >= threshold and label != "none":
                text = f"Tahmin: {label} ({confidence_val*100:.1f}%)"
                color = (0, 255, 0)
            else:
                text = "Tahmin: Emin degilim"
                color = (0, 0, 255)

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Canlı Kamera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



        
#%%main program
if __name__=="__main__":
    class_names = ["bulldog","pit bull","hound","beagle"]
    run_camera(model, class_names)
   


