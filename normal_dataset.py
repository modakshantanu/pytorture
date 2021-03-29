import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

def preprocess_1(dataset, fileName): 
    file = open(fileName, "r")
    
    samples = int(file.readline())
    dataset.total_samples += samples

    for i in range(samples):
        # print(i)
        time_samples = int(file.readline())
        
        cur_data = []

        for j in range(time_samples):
            fl = file.readline()
            part1 = list(map(int, fl.split(",")[10:13]))
            part2 = list(map(int, fl.split(",")[16:18]))
            
            part1 = [e * 1/2048 for e in part1]
            part2 = [e * 1/900 - 0.5 for e in part2]

            row = part1 + part2
            # row = [0,0, int(file.readline().split(",")[13])]        
            cur_data.append(row)
        
        # print(cur_data)
        cur_data = torch.FloatTensor(cur_data)
        # cur_data *= 1/2048
        cur_data = torch.transpose(cur_data, 0,1)

        for j in range(time_samples - 40 + 1):
            subrange = cur_data[:, j:j+40].clone()

            # Remove mean for acc values
            for k in range(3):
                subrange[k] -= torch.mean(subrange[k])

            # subrange = torch.transpose(subrange,0,1)
            # Augment data
            for k in range(10):
                dataset.data[dataset.data_idx] =  torch.unsqueeze(subrange + torch.rand_like(subrange) * 0.5e-2, dim=0)
                dataset.data_idx+=1
                # dataset.data[data_idx] = torch.flip(dataset.data[data_idx - 1], [1])
                # data_idx+=1


    file.close()



class NormDataset(Dataset):
    def __init__(self, files,  label = 2, dir = "data/capstone/dances/", transform=None, target_transform=None):
        
        self.total_samples = 0
        
        self.label = label
        self.data = torch.empty(500000, 5, 40)
        self.data_idx = 0

        for fileName in files:
            preprocess_1(self, dir + fileName)
            
        
        # self.data = self.data[torch.randperm(self.data.size()[0])]
        # if self.data_idx > 10000:
        #      self.data_idx = 10000
        self.data = self.data[:self.data_idx]
        print(self.data.shape)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data_idx

    def __getitem__(self, idx):

        # print(idx, self.data[idx], self.labels[idx])
        sample = (self.data[idx],  self.label)
        return sample