import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

from lab.dataset import mnist

# ---------- Reprodutibilidade ----------
torch.manual_seed(0)
np.random.seed(0)

# ---------- Dados ----------
X, y = mnist(1000)                 # 1000 amostras (64 features)
X = StandardScaler().fit_transform(X)  # padroniza (ajuda o treino)

X_tensor = torch.FloatTensor(X)
dataset  = TensorDataset(X_tensor, X_tensor)  # AE usa entrada como alvo
loader   = DataLoader(dataset, batch_size=64, shuffle=True)

# ---------- Modelo ----------
class Autoencoder(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()
        act = activation()
        self.encoder = nn.Sequential(
            nn.Linear(64, 60), act,
            nn.Linear(60, 30), act,
            nn.Linear(30, 2)             # gargalo 2D p/ plot
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 30), act,
            nn.Linear(30, 60), act,
            nn.Linear(60, 64)
        )
    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return z, xhat

def treinar(model, epochs, lr):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            _, xhat = model(xb)
            loss = crit(xhat, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        z, _ = model(X_tensor)
    return z.numpy()

# ---------- Configurações ----------
# RUIM: ativação Sigmoid (satura), lr alto, poucas épocas
ae_ruim   = Autoencoder(activation=nn.Sigmoid)
emb_ruim  = treinar(ae_ruim, epochs=5, lr=5e-2)

# MELHOR: ReLU, lr menor, mais épocas
ae_melhor  = Autoencoder(activation=nn.ReLU)
emb_melhor = treinar(ae_melhor, epochs=80, lr=1e-3)

# t-SNE direto do espaço original (64D) padronizado
tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=0)
emb_tsne = tsne.fit_transform(X)

# ---------- Uma única figura com 3 plots ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cm = "tab10"

axes[0].scatter(emb_ruim[:,0], emb_ruim[:,1], c=y, cmap=cm, s=10, alpha=0.7)
axes[0].set_title("RUIM (Sigmoid, lr=0.05, 5 épocas)")

axes[1].scatter(emb_melhor[:,0], emb_melhor[:,1], c=y, cmap=cm, s=10, alpha=0.7)
axes[1].set_title("MELHOR (ReLU, lr=0.001, 80 épocas)")

axes[2].scatter(emb_tsne[:,0], emb_tsne[:,1], c=y, cmap=cm, s=10, alpha=0.7)
axes[2].set_title("t-SNE (64D → 2D)")

for ax in axes:
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
fig.colorbar(plt.cm.ScalarMappable(cmap=cm), ax=axes, ticks=range(len(np.unique(y))), label="Dígito")
plt.tight_layout()
plt.show()
