import serial
import torch
import copy
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lrdataset import LRDataset
from normal_dataset import NormDataset
from mixed_dataset import MixedDataset
from extended_dataset import ExtDataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
class CNN(nn.Module):
    def __init__(self, in_channels = 16, output_size = 9):
        super(CNN, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16,kernel_size=5, stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(320, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
    def forward(self, x):
        x = self.stack(x)
        return x


model = CNN()
model.load_state_dict(torch.load("models/abc_1357.pth"))


def run_model(data):
    data = copy.deepcopy(data)
    ch_avg = [0] * 16
    # print(data[39])
    for i in range(40):
        data[i][0] /= 2048    
        data[i][1] /= 2048    
        data[i][2] /= 2048    
        data[i][3] /= 2048    
        data[i][4] /= 2048    
        data[i][5] /= 2048    
        data[i][6] /= 10000    
        data[i][7] /= 10000    
        data[i][8] /= 10000   
        data[i][9] /= 10000   
        data[i][10] /= 10000    
        data[i][11] /= 10000    
        data[i][12] /= 900    
        data[i][13] /= 900    
        data[i][14] /= 900    
        data[i][15] /= 900    

    for i in range(40):   
        for j in range(16):
            ch_avg[j] += data[i][j]
    
    for i in range(16):
        ch_avg[i] /= 40

    for j in range(40):

        for i in range(6):
            data[j][i] -= ch_avg[i]

    # print(data[0])
    data = torch.FloatTensor(data)
    data = torch.transpose(data, 0,1)
    data.unsqueeze_(0)
    score = model(data)

    _,prediction = score.max(1)

    return int(prediction)


buffer = []

danceNames = ["hair", "gun", "sidepump","pointhigh","wipetable","listen","dab","elbowkick","logout"]

port1 = serial.Serial('COM7', baudrate=115200, timeout=1)
port2 = serial.Serial('COM6', baudrate=115200, timeout=1)

while True:
    line1 = port1.readline()
    line2 = port2.readline()
    line1 = line1.decode(encoding='utf8')
    line2 = line2.decode(encoding='utf8')
    if len(line1) < 10 or len(line2) < 10: continue
    line1 = list(map(int, line1.split(",")))
    line2 = list(map(int, line2.split(",")))
    if line1[0] % 2 == 1:
        line1 , line2 = line2, line1

    res = line1[3:11] + line2[3:11]
    res = res[0:3] + res[8:11] + res[3:6] + res[11:14] + res[6:8] + res[14:16]
    # res = [0] * 16
    buffer.append(res)
    # print(buffer)
    if (len(buffer) == 40):
        move = run_model(buffer)
        print(danceNames[move])
        buffer = buffer[1:40]

