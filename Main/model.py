import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.utils.data import DataLoader

device = 'cuda' if (torch.cuda.is_available) else 'cpu'

EMBEDDING_DIM = 768
HIDDEN_DIM = 100

#example
#sentences = ["i like fresh bread", "i hate stale bread"]
#split into set -> {i, like, hate, fresh, stale, bread}
#index sentences dim -> (2,4) 
#arbitrary embedding (2,4,10)
#embed dim = 10
#hidden dim = 64
#lstm input shape (batch_size, seq_len, features) when batch_first=True
#lstm output shape (batch_size, seq_len, hidden_size)
#lstm(embed dim, hidden dim, )

#Temp embedding, word2vec
train_set = pd.read_csv("../Data/train.csv")
train_set.shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #BERT Embedding -> LSTM -> Linear

        # width ~ 40
        self.lstm_1 = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM) #(embedding dim, hidden dim)
        self.dropout_1 = nn.Dropout(0.2, inplace=True)
        self.lstm_2 = nn.LSTM()
        self.linear_1 = nn.Linear(HIDDEN_DIM, 2)


    def forward(self,x):
        y = self.lstm_1(x)
        y = self.linear_1(y)

        return y

model = Net()
print(model)



