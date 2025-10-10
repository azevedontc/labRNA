import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from lab.dataset import mnist

# ====== Função utilitária ======
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid)
    with torch.no_grad():
        Z = model(grid_tensor).numpy().reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
    plt.title(title)
    plt.show()

# ====== Dados ======
torch.manual_seed(0)
X, y = mnist(1800)
b = np.logical_or(y == 0, y == 1)
X, y = X[b], y[b]
y = (y == 1).astype(float)

# PCA → 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_pca)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0
)
X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1)
X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# ====== Modelo ======
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ====== Treino com função de perda genérica ======
def train_model(loss_fn_name):
    model = Classifier()
    torch.manual_seed(0)  # mesma semente p/ pesos iguais
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if loss_fn_name == "MSE":
        criterion = nn.MSELoss()
    elif loss_fn_name == "BCE":
        criterion = nn.BCELoss()  # entropia cruzada binária

    for epoch in range(100):
        for data, target in train_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Avaliação
    with torch.no_grad():
        preds = (model(X_test) > 0.5).float()
        acc = (preds == y_test).float().mean().item()
    return model, acc

# ====== Rodar os dois ======
model_mse, acc_mse = train_model("MSE")
model_bce, acc_bce = train_model("BCE")

print(f"Acurácia (MSE): {acc_mse:.3f}")
print(f"Acurácia (Entropia cruzada): {acc_bce:.3f}")

# ====== Plot ======
plot_decision_boundary(X_train.numpy(), y_train.numpy().flatten(), model_mse, "Fronteira - MSELoss")
plot_decision_boundary(X_train.numpy(), y_train.numpy().flatten(), model_bce, "Fronteira - Entropia Cruzada (BCE)")
