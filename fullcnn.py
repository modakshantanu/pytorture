# Imports
import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from lrdataset import LRDataset
from extended_dataset import ExtDataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datetime import datetime as dt

# create network
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 9
learning_rate = 0.001
num_epochs = 100

model = CNN(output_size=num_classes).to(device)



# Load data
# lr_data = LRDataset(["leftright_test.csv", "leftright_train.csv", "leftright_combined.csv"])#, "leftright_1.csv", "leftright_2.csv","leftright_3.csv"])
dance_data = torch.utils.data.ConcatDataset([

    # ExtDataset(files=["standing_combined.csv"], label=2),
    ExtDataset(files=["dab_combined.csv"], label=6),
    ExtDataset(files=["hair_combined.csv"], label=0),
    ExtDataset(files=["gun_combined.csv"], label=1),
    ExtDataset(files=["elbowkick_combined.csv"], label=7),
    ExtDataset(files=["listen_combined.csv"], label=5),
    ExtDataset(files=["pointhigh_combined.csv"], label=3),
    ExtDataset(files=["sidepump_combined.csv"], label=2),
    ExtDataset(files=["wipetable_combined.csv"], label=4),
    ExtDataset(files=["logout_1.csv"], label=8),
    ExtDataset(files=["logout_2.csv"], label=8),
    
    ExtDataset(files=["wipetable_combined.csv"], dir="data/capstone/24Mar/", label=4),
    ExtDataset(files=["sidepump_combined.csv"], dir="data/capstone/24Mar/", label=2),
    ExtDataset(files=["pointhigh_combined.csv"], dir="data/capstone/24Mar/", label=3),
    ExtDataset(files=["listen_combined.csv"], dir="data/capstone/24Mar/", label=5),
    ExtDataset(files=["hair_combined.csv"], dir="data/capstone/24Mar/", label=0),
    ExtDataset(files=["gun_combined.csv"], dir="data/capstone/24Mar/", label=1),
    ExtDataset(files=["elbowkick_combined.csv"], dir="data/capstone/24Mar/", label=7),
    ExtDataset(files=["dab_combined.csv"], dir="data/capstone/24Mar/", label=6),
])
    
    
all_data = torch.utils.data.ConcatDataset([
    # lr_data,
    dance_data
])

# lr_data = torch.utils.data.ConcatDataset([lr_data,  NormDataset(files=["standing_combined.csv"], label=2)])
    

total_samples = all_data.__len__()
train_cnt = round(total_samples * 0.8)
test_cnt = total_samples - train_cnt

train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_cnt, test_cnt])

print(f"Training samples: {train_dataset.__len__()}, Testing samples: {test_dataset.__len__()}")

# train_dataset = LRDataset(["leftright_train.csv"]) 
train_loader = DataLoader(dataset=train_dataset, batch_size = 64, shuffle=True)
# test_dataset = LRDataset(["leftright_test.csv"])
test_loader = DataLoader(dataset=test_dataset, batch_size = 64, shuffle=True)
dance_loader = DataLoader(dataset=dance_data, batch_size = 64, shuffle=True)
# lr_loader = DataLoader(dataset=lr_data, batch_size = 64, shuffle=True)


# Initialize network

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
            

            for i in range(len(predictions)):
                # print(f"actual = {int(y[i])} , prediction = {int(predictions[i])}")
                confMatrix[int(y[i])][int(predictions[i])]+=1
            num_correct += (predictions == y).sum()
            total_num += predictions.size(0)
    
    model.train()
    acc = num_correct / total_num * 100
    
    if showConf:
        for i in confMatrix:
            rowsum = sum(i)
            if rowsum == 0:
                rowsum += 1
            for j in i:
                print(f"{j / rowsum * 100 :.4f}", end="\t")
            print()

    # print(confMatrix)


    return acc



loss_record = []
lr_adjustments = 3

# Train Network

def train_loop():
    for epoch in range(num_epochs):
        global optimizer
        global criterion
        global loss_record
        global lr_adjustments
        global learning_rate
        total = 0
        correct = 0
        loss_amt = 0
        for batch, (input, answer) in enumerate(train_loader):
            # print(input, answer)
            input = input.to(device)
            answer = answer.to(device)

            # input = torch.flatten(input, start_dim=1)

            scores = model(input)
            loss = criterion(scores, answer)

            optimizer.zero_grad()
            loss.backward()
            loss_amt += loss.item()

            optimizer.step()
            total += input.shape[0]
            _, predictions = scores.max(1)

            correct += (predictions == answer).sum() 

        loss_record.append(loss_amt)
        print(f"Epoch {epoch}, accuracy = {correct / total * 100:.5f}, loss={loss_amt}")

        if epoch % 5 == 0:
            model.eval()
            tsa = check_accuracy(test_loader, model, True)
            # lr_acc = check_accuracy(lr_loader, model)
            print(f"Test accuracy = {tsa:.5f}")
            # print(f"LR accuracy = {lr_acc:.5f}")
            model.train()

        if (len(loss_record) > 5 and loss_amt / loss_record[-5] > 0.8):
            learning_rate /= 2
            print(f"Adjusted learning rate to {learning_rate}")
            lr_adjustments -= 1
            if lr_adjustments == 0:
                break
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def cleanup():
    # tra = check_accuracy(train_loader, model)
    # tsa = check_accuracy(test_loader, model)

    # print(f"Training accuracy = {tra:.5f}")
    # print(f"Test accuracy = {tsa:.5f}")


    torch.save(model.state_dict(), f"models/abc_{dt.now().hour}{dt.now().minute}.pth")

try:
    train_loop()    
except KeyboardInterrupt:
    print("Early termination")
    cleanup()

cleanup()
