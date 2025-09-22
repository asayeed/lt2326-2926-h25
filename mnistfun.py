import os
import glob
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.optim as optim

class MNISTDataset(Dataset):
    def __init__(self, location, class1, class2):
        super(MNISTDataset, self).__init__()
        class1glob = glob.glob(os.path.join(location, str(class1), "*.png"))
        class2glob = glob.glob(os.path.join(location, str(class2), "*.png"))
        transToTensor = transforms.ToTensor()
        
        imagetensors1 = [(transToTensor(Image.open(x)),0) for x in class1glob]
        imagetensors2 = [(transToTensor(Image.open(x)),1) for x in class2glob]

        self.imagetensors = imagetensors1 + imagetensors2

    def __len__(self):
        return len(self.imagetensors)

    def __getitem__(self, x):
        return self.imagetensors[x]

class DigitClassificationModel(nn.Module):
    def __init__(self):
        super(DigitClassificationModel, self).__init__()
        self.conv2d = nn.Conv2d(1, 1, (2,2), padding="same")
        self.flatten = nn.Flatten(1,2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(784, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, item):
        output = self.conv2d(item)
        output = self.flatten(output)
        output = self.relu(output)
        output = self.linear(output)
        output = self.sigmoid(output)

        return output

def train(dataset, batch_size=25, epochs=3):
    dataloader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)
    model = DigitClassificationModel()
    optimizer = optim.Adam
    for epoch in epochs:
        for batch_index, sample in enumerate(dataloader):
            X, y = sample
            output = model(X)
            