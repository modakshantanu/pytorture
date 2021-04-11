import torch
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CNN().to(device)

model.load_state_dict(torch.load("models/abc_49178.pth"))
num_classes  =9

test_data = torch.utils.data.ConcatDataset([

    # ExtDataset(files=["standing_combined.csv"], label=2),
    # ExtDataset(files=["dab_combined.csv"], label=6),
    # ExtDataset(files=["hair_combined.csv"], label=0),
    # ExtDataset(files=["gun_combined.csv"], label=1),
    # ExtDataset(files=["elbowkick_combined.csv"], label=7),
    # ExtDataset(files=["listen_combined.csv"], label=5),
    # ExtDataset(files=["pointhigh_combined.csv"], label=3),
    # ExtDataset(files=["sidepump_combined.csv"], label=2),
    # ExtDataset(files=["wipetable_combined.csv"], label=4),
    # ExtDataset(files=["logout_1.csv"], label=8),
    # ExtDataset(files=["logout_2.csv"], label=8),
    
    # ExtDataset(files=["wipetable_combined.csv"], dir="data/capstone/24Mar/", label=4),
    # ExtDataset(files=["sidepump_combined.csv"], dir="data/capstone/24Mar/", label=2),
    # ExtDataset(files=["pointhigh_combined.csv"], dir="data/capstone/24Mar/", label=3),
    # ExtDataset(files=["listen_combined.csv"], dir="data/capstone/24Mar/", label=5),
    # ExtDataset(files=["hair_combined.csv"], dir="data/capstone/24Mar/", label=0),
    # ExtDataset(files=["gun_combined.csv"], dir="data/capstone/24Mar/", label=1),
    # ExtDataset(files=["elbowkick_combined.csv"], dir="data/capstone/24Mar/", label=7),
    # ExtDataset(files=["dab_combined.csv"], dir="data/capstone/24Mar/", label=6),
    MixedDataset(files=["mixed_4916443.csv"])
])

test_loader = DataLoader(dataset=test_data, batch_size = 64, shuffle=True)

# Check accuracy
def check_accuracy(loader, model, showConf=False):
    num_correct = 0
    total_num = 0
    model.eval()

    confMatrix = [[0] * num_classes for i in range(num_classes)]
    # confMatrix = torch.tensor(num_classes, num_classes)

    with torch.no_grad():
        for x , y in loader:
            x = x.to(device)
            y = y.to(device)
            # x = torch.flatten(x, start_dim=1)
            scores = model(x)
            _, predictions = scores.max(1)
            # print(y, predictions)


                
            for i in range(len(predictions)):
                # print(f"actual = {int(y[i])} , prediction = {int(predictions[i])}")
                confMatrix[int(y[i])][int(predictions[i])]+=1
                total_num+=1
                if (int(y[i]) == int(predictions[i])):
                    num_correct+=1
    
    acc = num_correct / total_num * 100
    
    if showConf:
        for i in confMatrix:
            rowsum = sum(i)
            if rowsum == 0:
                rowsum += 1
            for j in i:
                print(f"{j / rowsum * 100 :.3f}", end="\t")
            print()

    # print(confMatrix)


    return acc


tsa = check_accuracy(test_loader, model, True)
# lr_acc = check_accuracy(lr_loader, model)
print(f"Test accuracy = {tsa:.5f}")