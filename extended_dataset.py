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
            values = list(map(int, fl.split(",")[1:9]))
            values = [0] + values + [0]
            values += list(map(int, fl.split(",")[10:]))
            acc = values[1:4] + values[10:13]
            gyro = values[4:7] + values[13:16]
            pr = values[7:9] + values[16:18]

            acc = [e * 1/2048 for e in acc]
            gyro = [e * 1/10000 for e in gyro]
            pr = [(e // 10) * 1/90 for e in pr]

            row = acc + gyro + pr
            # row = [0,0, int(file.readline().split(",")[13])]        
            cur_data.append(row)
        
        # print(cur_data)
        cur_data = torch.FloatTensor(cur_data)
        # cur_data *= 1/2048
        cur_data = torch.transpose(cur_data, 0,1)

        for j in range(time_samples - 40 + 1):
            subrange = cur_data[:, j:j+40].clone()

            # Remove mean for acc values
            for k in range(6):
                subrange[k] -= torch.mean(subrange[k])

            # subrange = torch.transpose(subrange,0,1)
           
            # Augment data
            for k in range(1):
                dataset.data[dataset.data_idx] =  torch.unsqueeze(subrange + torch.randn_like(subrange) * 0.0, dim=0)
                dataset.data_idx+=1
                # dataset.data[data_idx] = torch.flip(dataset.data[data_idx - 1], [1])
                # data_idx+=1


    file.close()



class ExtDataset(Dataset):
    def __init__(self, files,  label = 2, dir = "data/capstone/dances/", transform=None, target_transform=None):
        
        self.total_samples = 0
        
        self.label = label
        self.data = torch.empty(500000, 16, 40)
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