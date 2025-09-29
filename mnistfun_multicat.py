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
    def __init__(self, location, classes):
        super(MNISTDataset, self).__init__()
        self.classes = classes
        self.classindex = list(range(len(classes)))
        self.imagetensors = []
        transToTensor = transforms.ToTensor()
        
        for i in self.classindex:
            classglob = glob.glob(os.path.join(location, str(self.classes[i]), "*.png"))        
            imagetensors = [(transToTensor(Image.open(x)),i) for x in classglob]
            self.imagetensors += imagetensors

    def __len__(self):
        return len(self.imagetensors)

    def __getitem__(self, x):
        return self.imagetensors[x]

class DigitClassificationModel(nn.Module):
    def __init__(self, classes=2):
        super(DigitClassificationModel, self).__init__()
        self.conv2d = nn.Conv2d(1, 1, (2,2), padding="same")
        self.flatten = nn.Flatten(2,3)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(784, classes)
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

def train(dataset, classes, batch_size=25, epochs=3):
    dataloader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)
    model = DigitClassificationModel(classes)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()
    for epoch in range(epochs):
        total_loss = 0
        for batch_index, sample in enumerate(tqdm.tqdm(dataloader)):
            optimizer.zero_grad()
            X, y = sample
            output = model(X)
            
            #print(f"dtypes y {y.dtype} output {output.dtype}")
            #print(f"y size {y.size()} output size {output.size()}")
            #print(f"output {output} y {y}")
            loss = criterion(torch.squeeze(output), y)
            loss.backward()
            total_loss += float(loss)
            optimizer.step()
        print(f"Total loss = {total_loss}")

    return model
            
            
