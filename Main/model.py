import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
EMBEDDING_DIM = 768
HIDDEN_DIM = 10    # might change to smaller number

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 1):
        super(Model, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigm = nn.Sigmoid()
        
    # create function to init state
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
        
    
    def forward(self, x):     
        batch_size = x.size(0)
        h = self.init_hidden(batch_size).to(device)
        
        out, h = self.rnn(x, h)       
        out = self.fc(out)
        out = self.sigm(out)
        
        #return out, h
        return out
        

### TRAIN
def train_model(dataloader, model, criterion, optimizer, device, num_epochs = 25):
    model.train()
    correct = 0
    numSamples = 0

    for epoch in range(num_epochs):
        train_loss = 0
        
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            yhat = model(x)
            loss = loss_func(yhat, y)
            # y = torch.Tensor([1 if x > 0.5 else 0 for x in yhat])
            y_predict = torch.round(yhat)
            correct += torch.sum(y_predict == y)
            numSamples += len(y)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss
        
        
        print(f'Epoch: {epoch+1:02d}, ' +
            f'Loss: {train_loss / len(train_dataloader.dataset):.4f}' + 
            f'accuracy: '{correct / numSamples})

    print('Finished Training')

if __name__ == '__main__':
    device = 'cuda' if (torch.cuda.is_available) else 'cpu'

    model = Model(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, output_size=1)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(dataloader, model, criterion, optimizer, device, num_epochs = 25)



