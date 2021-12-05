import torch
import torch.nn as nn
import torchvision
import numpy as np
import pickle

from torch.utils.data import DataLoader, Dataset

EMBEDDING_DIM = 768
HIDDEN_DIM = 100    # might change to smaller number
Train_DIR1 = "/Users/anfen/Documents/EE541_final_data"
Train_DIR2  = "./EE541_FINAL_PROJ/Main/"
Test_DIR = ""


def create_dataset_train():
    with open(Train_DIR1 + "/dict.pkl", 'rb') as f:
        data = pickle.load(f)

    with open(Train_DIR1 + '/ids.pkl', 'rb') as f:
        ids = pickle.load(f)
    embeddings = []
    labels = []
    for id in ids:
        embeddings.append(torch.Tensor(data[id]['embedding']))
        labels.append(torch.Tensor(int(data[id]['target'])))
    
    return embeddings, labels

def create_dataset_test():
    with open(Test_DIR1 + "/dict.pkl", 'rb') as f:
        data = pickle.load(f)

    with open(Test_DIR1 + '/ids.pkl', 'rb') as f:
        ids = pickle.load(f)
    embeddings = []
    labels = []
    for id in ids:
        embeddings.append(data[ids]['embedding'])
        labels.append(int(data[ids]['target']))
    
    return embeddings, labels

class custom_Dataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem_(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]

        #embedding 50x768 and target (0x1)
        sample = {"Embedding": embedding, "Label": label}
        return sample

def get_dataloader(batch_size, num_workers):
    X_train, y_train = create_dataset_train()
    # X_test, y_test = create_dataset_test()

    train_set = custom_Dataset(X_train, y_train)
    # test_set = custom_Dataset(X_test, y_test)
    # dataset_size = {'train': len(y_train), 'test': len(y_test)}

    # datasets = {'train': train_set, 'test': test_set}
    # dataloaders = {x: DataLoader(datasets[x],
    #                              shuffle=True if x=='train' else False,
    #                              batch_size=batch_size,
    #                              num_workers=num_workers)
    #                              for x in ['train', 'test']}
    dataset_size = {'train': len(y_train)}
    datasets = {'train': train_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train']}
    return dataloaders, dataset_size

if __name__ == '__main__':
    get_dataloader(64, 4)