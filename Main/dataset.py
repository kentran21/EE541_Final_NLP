import torch
import torch.nn as nn
import torchvision
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

Train_DIR  = "../Data"
Test_DIR = "../Data"

def create_dataset_train_val():
    with open(Train_DIR + "/dict_train.pkl", 'rb') as f:
        data = pickle.load(f)

    with open(Train_DIR + '/ids_train.pkl', 'rb') as f:
        ids = pickle.load(f)
    embeddings = []
    labels = []
    for _id in ids:
        embeddings.append(torch.Tensor(data[_id]['embedding']))
        labels.append(torch.Tensor([int(data[_id]['target'])]))
    X_train, X_val, y_train , y_val = train_test_split(embeddings, labels, test_size=0.2)
    return X_train, X_val, y_train , y_val

def create_dataset_test():
    with open(Test_DIR + "/dict_test.pkl", 'rb') as f:
        data = pickle.load(f)

    with open(Test_DIR + '/ids_test.pkl', 'rb') as f:
        ids = pickle.load(f)
    embeddings = []
    for _id in ids:
        embeddings.append(torch.Tensor(data[_id]['embedding']))
    
    return embeddings, ids

class train_val_Dataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]

        #embedding 50x768 and target (0x1)
        return embedding, label

class test_Dataset(Dataset):
    def __init__(self, embeddings, ids):
        self.embeddings = embeddings
        self.ids = ids
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        _id = self.ids[idx]

        # embedding 50x768 and target (0x1)
        return embedding, _id

def get_dataloader(batch_size, num_workers):
    X_train, X_val, y_train , y_val = create_dataset_train_val() 
    X_test, ids_test = create_dataset_test()

    train_set = train_val_Dataset(X_train, y_train)
    val_set = train_val_Dataset(X_val, y_val)
    test_set = test_Dataset(X_test, ids_test)

    dataset_size = {'train': len(y_train), 'val': len(y_val), 'test': len(ids_test)}
    datasets = {'train': train_set, 'val': val_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=1 if x=='test' else batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'val', 'test']}
     
    return dataloaders, dataset_size

if __name__ == '__main__':
    get_dataloader(64, 4)