# practice MLP


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate synthetic dataset -------------------------------------
# Suppose we have 3 features: [size (m^2), number of rooms, age (years)]
np.random.seed(42)
n_samples = 1000

size = np.random.uniform(50, 200, n_samples)
rooms = np.random.randint(1, 6, n_samples)
age = np.random.uniform(0, 50, n_samples)

# Target: price = 50k + 300*size + 10k*rooms - 500*age + noise
price = 50_000 + 300*size + 10_000*rooms - 500*age + np.random.normal(0, 10_000, n_samples)

X = np.vstack([size, rooms, age]).T
y = price

# Normalize input features (important for NN training)
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X = (X - X_mean) / X_std

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


# 2. Build a custom Dataset class ----------------------------------
class HouseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = HouseDataset(X_tensor, y_tensor)

# Split into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


# 3. Define the MLP model -----------------------------------------
class MLPRegression(nn.Module):
    def __init__(self, in_features, hidden_sizes=[64, 64], dropout=0.1):
        super().__init__()
        layers = []
        last_size = in_features
        for hs in hidden_sizes:
            layers += [
                nn.Linear(last_size, hs),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            last_size = hs
        layers.append(nn.Linear(last_size, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = MLPRegression(in_features=3)
print(model)


# 4. Define loss and optimizer -------------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 5. Train the model -----------------------------------------------
n_epochs = 100
train_losses, test_losses = [], []

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_train_loss = total_loss / len(train_loader.dataset)

    # Evaluate on test set
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    avg_test_loss = total_loss / len(test_loader.dataset)

    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.2f} | Test Loss: {avg_test_loss:.2f}")


# 6. Evaluate and visualize ----------------------------------------
model.eval()
with torch.no_grad():
    X_test, y_test = test_set[:][0], test_set[:][1]
    y_pred = model(X_test).squeeze().numpy()
    y_true = y_test.squeeze().numpy()

plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("House Price Regression: True vs Predicted")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.show()

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.title("Training Curve (MSE Loss)")
plt.show()
