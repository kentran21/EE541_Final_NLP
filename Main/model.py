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
import os.path as osp
import csv
import copy

from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score

EMBEDDING_DIM = 768
HIDDEN_DIM = 50   
NUM_LAYERS = 2
MODEL_PATH = "../DATA/"

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, dropout=0, output_size=1):
        super(Model, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.dropout = dropout
        

        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.layer_size, batch_first=True, dropout = 0.8, bidirectional = True)
        self.fc = nn.Linear(self.hidden_size*2, self.output_size)
        self.sigm = nn.Sigmoid()
        
    # create function to init state
    def init_hidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size)

    # for LSTM
    def init_cell(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
    
    def forward(self, x):     
        batch_size = x.size(0)
        h = self.init_hidden(batch_size).to(device)
        c = self.init_cell(batch_size).to(device)
        
        out, h = self.rnn(x, h) # for GRU
        out = self.fc(out[:, -1, :])
        out = self.sigm(out)
        
        #return out
        return out
        

### TRAIN
def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):
    best_acc = 0.0
    train_loss_sv = []
    val_loss_sv = []
    train_acc_sv = []
    val_acc_sv = []

    for epoch in tqdm(range(num_epochs)):
        for phase in ['train', 'val']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_corrects = 0
            y_save = []
            ypred_save = []

            for x, y in tqdm(dataloader[phase]):
                x = torch.transpose(x,1,2)
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    yhat = model(x)

                    loss = criterion(yhat, y)
                    y_predict = torch.round(yhat)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                y_save.extend(y.detach().cpu().tolist())
                ypred_save.extend(y_predict.detach().cpu().tolist())
                #template for saving model
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(y_predict==y)
            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]
            epoch_acc = epoch_acc.detach().cpu().numpy()
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'train':
                train_loss_sv.append(epoch_loss)
                train_acc_sv.append(epoch_acc)
            else:
                val_loss_sv.append(epoch_loss)
                val_acc_sv.append(epoch_acc)
            f1 = f1_score(y_save, ypred_save)
            print(f'Phase: {phase}, ' +
                f'Epoch: {epoch+1:02d}, ' +
                f'f1 score: {f1:02f}, ' +
                f'Loss: {running_loss / dataset_size[phase]:.4f}, ' + 
                f'accuracy: {running_corrects / dataset_size[phase]:.4f}')

    torch.save(best_model_wts, osp.join(MODEL_PATH, 'model.pth'))
    print('Model saved at: {}'.format(osp.join(MODEL_PATH, 'model.pth')))
            
        

    print('Finished Training')
    return train_loss_sv, val_loss_sv, train_acc_sv, val_acc_sv


def plot(train_loss, val_loss, train_acc, val_acc, title): 
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].plot(train_loss)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("loss")

    ax[0].plot(val_loss)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("loss")
    ax[0].legend(["train", "validation"])
    ax[0].grid()

    ax[1].plot(train_acc)
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("accuracy")

    ax[1].plot(val_acc)
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("accuracy")
    ax[0].legend(["train", "validation"])
    ax[1].grid()
    plt.show()

def test_model(dataloader, model, device, dataset_size):

    ids_all = []
    predictions_all = []
    for phase in ['test']:
        for x, ids in tqdm(dataloader[phase]):
            x = torch.transpose(x,1,2)
            x = x.to(device)

            with torch.set_grad_enabled(False):
                yhat = model(x)
                y_predict = torch.round(yhat).flatten().tolist()
            #y_predict = y_predict.detach().cpu().numpy()
            ids_all.extend(ids)
            predictions_all.extend(y_predict)
    print(predictions_all)
    header = ['id', 'target']
    result = list(map(list, zip(ids_all, predictions_all)))

    temp = []
    for i in predictions_all:
        temp.append(i)

    for i in temp:
        i = int(i)

    with open('submission.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(result)




if __name__ == '__main__':
    device = 'cuda' if (torch.cuda.is_available) else 'cpu'

    dataloaders, dataset_size = get_dataloader(128,6)

    val = input("train data or test data, enter 'train' or 'test': ")

    if val == 'train':
        model = Model(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, layer_size=NUM_LAYERS, output_size=1)
        print(model)
        model.to(device)
        
        criterion = nn.BCELoss()
        learning_rate = 0.001
        num_epochs = 10
        title = f"GRU Bidirectional,  lr = {learning_rate}, hidden_size = {HIDDEN_DIM}, num_layers = {NUM_LAYERS}"
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.001)
        train_loss, val_loss, train_acc, val_acc = train_model(dataloaders, model, criterion, optimizer, device, num_epochs, dataset_size=dataset_size)

        print(train_loss, val_loss, train_acc, val_acc)
        plot(train_loss, val_loss, train_acc, val_acc, title)
    elif val == 'test':
        model = Model(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, layer_size=NUM_LAYERS, output_size=1)
        model.load_state_dict(torch.load(osp.join(MODEL_PATH, 'model.pth')))
        model.to(device)
        model.eval()
        test_model(dataloaders, model, device, dataset_size)


