from dataset import get_dataloader
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
from tqdm import tqdm

EMBEDDING_DIM = 768
HIDDEN_DIM = 400   

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 1):
        super(Model, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigm = nn.Sigmoid()
        
    # create function to init state
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
        
    
    def forward(self, x):     
        batch_size = x.size(0)
        h = self.init_hidden(batch_size).to(device)
        
        out, h = self.rnn(x, h)
        # out = self.fc(out[:, 0, :])   maybe this is correct  
        out = self.fc(out[:, -1, :])
        out = self.sigm(out)
        
        #return out, h
        return out
        

### TRAIN
def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):
    model.train()
    # best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(num_epochs)):
        # for phase in ['train', 'val']:
        for phase in ['train']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_corrects = 0

            for x, y in tqdm(dataloader):
                x = torch.transpose(x,1,2)
                x = x.to(device)
                y = y.to(device)

                yhat = model(x)
                loss = criterion(yhat, y)
                # y = torch.Tensor([1 if x > 0.5 else 0 for x in yhat])
                y_predict = torch.round(yhat)

                model.zero_grad()
                loss.backward()
                optimizer.step()

                #template for saving model
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(y_predict==y)

            # epoch_loss = running_loss / dataset_size['phase']
            # epoch_acc = running_corrects.double() / dataset_size['phase']

            # if epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

                # torch.save(best_model_wts, osp.join(Config['./'], Config['./'], 'model.pth'))
                # print('Model saved at: {}'.format(osp.join(Config['./'], Config['./'], 'model.pth')))

            
            
            print(f'Epoch: {epoch+1:02d}, ' +
                f'Loss: {running_loss / len(dataloader.dataset):.4f}, ' + 
                f'accuracy: {running_corrects / dataset_size[phase]}')

    print('Finished Training')


def plot(train_loss, val_loss, train_acc, val_acc, title=""): 
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].plot(train_loss)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("loss")

    ax[0].plot(val_loss)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("loss")
    ax[0].legend(["train", "validation"])

    ax[1].plot(train_acc)
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("accuracy")

    ax[1].plot(val_acc)
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("accuracy")
    ax[0].legend(["train", "validation"])
    if title:
        fig.suptitle(title)
    plt.show()



if __name__ == '__main__':
    device = 'cuda' if (torch.cuda.is_available) else 'cpu'

    dataloaders, dataset_size = get_dataloader(64,6)
    dataloader = dataloaders['train']
    # dataloader_test = dataloaders['test']

    model = Model(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, output_size=1)
    print(model)
    model.to(device)
    criterion = nn.BCELoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(dataloader, model, criterion, optimizer, device, num_epochs = 25, dataset_size=dataset_size)



