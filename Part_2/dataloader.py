import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class MNISTDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1) / 255.0
        self.label = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

def get_loaders(train_fraction, batch_size, seed, full_data_return = False):
    np.random.seed(seed)

    train_data1 = np.load('./data/data0.npy')
    train_data2 = np.load('./data/data1.npy')
    train_data3 = np.load('./data/data2.npy')
    train_label1 = np.load('./data/lab0.npy')
    train_label2 = np.load('./data/lab1.npy')
    train_label3 = np.load('./data/lab2.npy')

    full_data = np.concatenate([train_data1, train_data2, train_data3])
    full_label = np.concatenate([train_label1, train_label2, train_label3])
    len_arr = np.arange(len(full_label))
    np.random.shuffle(len_arr)

    full_data = full_data[len_arr]
    full_label = full_label[len_arr]

    if full_data_return:
        full_dataset = MNISTDataset(full_data, full_label)
        full_loader = DataLoader(full_dataset, batch_size=batch_size)
        return full_loader

    train_len = int(len(full_label)*train_fraction)

    train_data = full_data[:train_len]
    train_label = full_label[:train_len]
    val_data = full_data[train_len:]
    val_label = full_label[train_len:]

    train_dataset = MNISTDataset(train_data, train_label)
    val_dataset = MNISTDataset(val_data, val_label)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader