# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    def __init__(self, in_channels = 1, output_size = 10):
        super(CNN, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Flatten(),
            nn.Linear(64*7*7, 250),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(250,output_size)
        )
    def forward(self, x):
        x = self.stack(x)
        return x



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28*28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 20

# Load data
train_dataset = datasets.FashionMNIST("/data", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.FashionMNIST("/data", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

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
    print(f"Epoch {epoch}, accuracy = {correct / total * 100:.2f} loss = {loss_amt}")
    tsa = check_accuracy(test_loader, model)
    print(f"Test accuracy = {tsa:.2f}")


tra = check_accuracy(train_loader, model)
tsa = check_accuracy(test_loader, model)

print(f"Training accuracy = {tra:.5f}")
print(f"Test accuracy = {tsa:.5f}")