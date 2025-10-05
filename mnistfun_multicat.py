import os
import glob
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import tqdm

class MNISTDataset(Dataset):
    def __init__(self, location, classes, device=None):
        super(MNISTDataset, self).__init__()
        self.classes = classes
        self.classindex = list(range(len(classes)))
        self.imagetensors = []
        transToTensor = transforms.ToTensor()
        
        for i in self.classindex:
            classglob = glob.glob(os.path.join(location, str(self.classes[i]), "*.png"))        
            imagetensors = [(transToTensor(Image.open(x)),i) for x in classglob]
            self.imagetensors += imagetensors

        if device:
            self.imagetensors = [(x[0].to(device), x[1]) for x in self.imagetensors]

    def __len__(self):
        return len(self.imagetensors)

    def __getitem__(self, x):
        return self.imagetensors[x]

class MNISTGenerationDataset(Dataset):
    def __init__(self, location, classes, device=None):
        super(MNISTDataset, self).__init__()
        self.classes = classes
        self.classindex = list(range(len(classes)))
        self.imagetensors = []
        transToTensor = transforms.ToTensor()
        
        for i in self.classindex:
            classglob = glob.glob(os.path.join(location, str(self.classes[i]), "*.png"))        
            imagetensors = [(transToTensor(Image.open(x)),i) for x in classglob]
            self.imagetensors += imagetensors

        if device:
            self.imagetensors = [(x[0].to(device), x[1]) for x in self.imagetensors]

    def __len__(self):
        return len(self.imagetensors)

    def __getitem__(self, x):
        return self.imagetensors[x]


class DigitClassificationModel(nn.Module):
    def __init__(self, classes=2, device='cpu'):
        super(DigitClassificationModel, self).__init__()
        self.conv2d = nn.Conv2d(1, 1, (2,2), padding="same", device=device)
        self.flatten = nn.Flatten(2,3)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(784, classes, device=device)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, item):
        #print(f"initial size {item.size()}")
        output = self.conv2d(item)
        #print(f"conv2d size {output.size()}")
        output = self.flatten(output)
        #print(f"flatten size {output.size()}")
        output = self.tanh(output)
        output = self.linear(output)
        output = self.logsoftmax(output)

        return output

class DigitGenerationModel(nn.Module):
    def __init__(self, classes=10, device='cpu'):
        super(DigitGenerationModel, self).__init__()
        self.linear = nn.Linear(classes, 784, device=device)
        self.tanh = nn.Tanh()
        self.unflatten = nn.Unflatten(1, (28,28))
        self.conv2d = nn.Conv2d(1, 1, (2, 2), padding="same", device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, item):
        output = self.linear(item)
        output = self.tanh(output)
        output = self.unflatten(output)
        output = self.conv2d(output)
        output = self.sigmoid(output)
        return output

def train(dataset, classes, batch_size=25, epochs=3, device='cpu'):
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    model = DigitClassificationModel(classes, device=device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()
    for epoch in range(epochs):
        total_loss = 0
        for batch_index, sample in enumerate(tqdm.tqdm(dataloader)):
            optimizer.zero_grad()
            X, y = sample
            output = model(X)
            y = y.to(device)
            #print(f"dtypes y {y.dtype} output {output.dtype}")
            #print(f"y size {y.size()} output size {output.size()}")
            #print(f"output {output} y {y}")
            loss = criterion(torch.squeeze(output), y)
            loss.backward()
            total_loss += float(loss)
            optimizer.step()
        print(f"Total loss = {total_loss} at epoch {epoch}.")

    return model
            
            
def train(dataset, classes, batch_size=25, epochs=3, device='cpu'):
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    model = DigitGenerationModel(classes, device=device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()          
    for epoch in range(epochs):
        total_loss = 0
        for batch_index, sample in enumerate(tqdm.tqdm(dataloader)):
            optimizer.zero_grad()
            X, y = sample
            y = y.to(device)

            output = model(X)
            #print(f"dtypes y {y.dtype} output {output.dtype}")
            #print(f"y size {y.size()} output size {output.size()}")
            #print(f"output {output} y {y}")
            loss = criterion(torch.squeeze(output), y)
            loss.backward()
            total_loss += float(loss)
            optimizer.step()
        print(f"Total loss = {total_loss} at epoch {epoch}.")

    return model
            
