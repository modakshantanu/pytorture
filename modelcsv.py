import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lrdataset import LRDataset
from normal_dataset import NormDataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CNN(nn.Module):
    def __init__(self, in_channels = 5, output_size = 2):
        super(CNN, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=8,kernel_size=5, stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(8*20, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )

        # self.stack = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(3 * 40, output_size)
        # )
    def forward(self, x):
        x = self.stack(x)
        return x
num_classes = 12

model = CNN(output_size=num_classes)

model.load_state_dict(torch.load("models/abc_1558.pth"))

wb = model.state_dict()

for i in wb:
    # print(i, wb[i].shape)
    wb[i] = torch.flatten(wb[i])
    # print(wb[i])
    for j in wb[i]:
        print(float(j))

