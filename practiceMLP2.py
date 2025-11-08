import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split





# 1. Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

size = np.random.uniform(50, 200, n_samples)
rooms = np.random.randint(1, 6, n_samples)
age = np.random.uniform(0, 50, n_samples)

price = 50_000 + 300*size + 10_000*rooms - 500*age + np.random.normal(0, 10000, n_samples)

X = np.vstack([size, rooms, age]).T
y = price

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X = (X-X_mean)/X_std

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)




# 2. Build a custom Dataset class 
class HouseDataset(Dataset):
    def __init__(self, X,y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
dataset = HouseDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)




# 3. Define the MLP model
class Fcnn(nn.Module):
    def __init__(self, in_features, hidden_sizes=[64,64], dropout=0.1):
        super().__init__()

        layers = []
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self,x):
        return self.net(x)
    

model = Fcnn(in_features=3)

# 4. Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Train the model
n_epoches = 100
train_loss, test_loss = [], []

for epoch in range(n_epoches):
    model.train()
    total_loss = 0

    for idx, (X_batch, y_batch) in enumerate(train_loader):
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()*X_batch.size(0)
    avg_train_loss = total_loss / len(train_loader.dataset)



    model.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, (X_batch, y_batch) in enumerate(test_loader):
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss = total_loss + loss.item()*X_batch.size(0)
    avg_test_loss = total_loss / len(test_loader.dataset)

    train_loss.append(avg_train_loss)
    test_loss.append(avg_test_loss)

    if (epoch+1) % 10 == 0:
        print("1")



model.eval()
with torch.no_grad():
    X_test, y_test = test_set[:][0], test_set[:][1]
    y_pred = model(X_test).squeeze().numpy()
    y_true = y_test.squeeze().numpy()




print("finish!")


nn.Conv2d(inc, outc, kernel_size=, padd)
nn.AvgPool2d()

