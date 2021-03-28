# Imports
import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lrdataset import LRDataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# create network
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)
        
        self.layer2 = nn.Linear(50, output_size)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# create network
class CNN(nn.Module):
    def __init__(self, in_channels = 3, output_size = 2):
        super(CNN, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=8,kernel_size=5, stride=1,padding=2),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(8*20, output_size)
        )

        # self.stack = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(3 * 40, output_size)
        # )
    def forward(self, x):
        x = self.stack(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 2
learning_rate = 0.001
batch_size = 50
num_epochs = 200

# Load data
all_data = LRDataset(["leftright_test.csv", "leftright_train.csv", "leftright_combined.csv"])

total_samples = all_data.__len__()
train_cnt = round(total_samples * 0.8)
test_cnt = total_samples - train_cnt

train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_cnt, test_cnt])

print(f"Training samples: {train_dataset.__len__()}, Testing samples: {test_dataset.__len__()}")

# train_dataset = LRDataset(["leftright_train.csv"]) 
train_loader = DataLoader(dataset=train_dataset, batch_size = 64, shuffle=True)
# test_dataset = LRDataset(["leftright_test.csv"])
test_loader = DataLoader(dataset=test_dataset, batch_size = 64, shuffle=True)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    total_num = 0
    model.eval()

    with torch.no_grad():
        for x , y in loader:
            x = x.to(device)
            y = y.to(device)
            # x = torch.flatten(x, start_dim=1)

            scores = model(x)
            _, predictions = scores.max(1)
            
            num_correct += (predictions == y).sum()
            total_num += predictions.size(0)
    
    model.train()
    acc = num_correct / total_num * 100
    
    return acc
# Train Network
for epoch in range(num_epochs):
    
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
    print(f"Epoch {epoch}, accuracy = {correct / total * 100:.2f}, loss={loss_amt}")
    model.eval()
    tsa = check_accuracy(test_loader, model)
    print(f"Test accuracy = {tsa:.2f}")
    model.train()


tra = check_accuracy(train_loader, model)
tsa = check_accuracy(test_loader, model)

print(f"Training accuracy = {tra:.2f}")
print(f"Test accuracy = {tsa:.2f}")