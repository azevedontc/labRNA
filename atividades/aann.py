import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine  # -> troque aqui se quiser outro dataset
# from sklearn.datasets import load_breast_cancer
# from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# Reprodutibilidade
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

data = load_wine()
X = data.data.astype(np.float32)  # (n amostras, 13 features)
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

X_tensor = torch.from_numpy(X_scaled)
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)

in_dim = X.shape[1]  # 13

class RNAAutoAssociativa(nn.Module):
    def __init__(self, in_dim, hidden=(8,), bottleneck=2, activation="relu"):
        super().__init__()
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "none": nn.Identity(),
        }

        enc_layers = []
        last = in_dim
        for h in hidden:
            enc_layers += [nn.Linear(last, h)]
            if activation != "none":
                enc_layers += [acts[activation]]
            last = h
        enc_layers += [nn.Linear(last, bottleneck)]  # gargalo 2D
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        last = bottleneck
        for h in reversed(hidden):
            dec_layers += [nn.Linear(last, h)]
            if activation != "none":
                dec_layers += [acts[activation]]
            last = h
        dec_layers += [nn.Linear(last, in_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return z, xrec

def treinar(ae, dataloader, epochs=50, lr=1e-3, weight_decay=0.0, noise_std=0.0, log_every=20):
    opt = optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    ae.train()
    for ep in range(epochs):
        for xb, _ in dataloader:
            xb_noisy = xb + noise_std * torch.randn_like(xb) if noise_std > 0 else xb
            z, xr = ae(xb_noisy)
            loss = crit(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % log_every == 0:
            print(f"Época {ep+1:3d}/{epochs}: MSE {loss.item():.4f}")

    ae.eval()
    with torch.no_grad():
        z, _ = ae(X_tensor)
    return z.numpy()

# 3) Configurações: RUIM x MELHOR

# RUIM: rede muito rasa, linear (sem não-linearidade), poucas épocas
ae_ruim = RNAAutoAssociativa(in_dim, hidden=(8,), bottleneck=2, activation="none")
emb_ruim = treinar(ae_ruim, dataloader, epochs=25, lr=0.005, weight_decay=0.0, noise_std=0.0, log_every=5)

# MELHOR: rede mais profunda + ReLU, mais épocas, leve regularização e ruído (denoising)
ae_bom = RNAAutoAssociativa(in_dim, hidden=(32, 16), bottleneck=2, activation="relu")
emb_bom = treinar(ae_bom, dataloader, epochs=200, lr=0.003, weight_decay=1e-4, noise_std=0.05, log_every=20)

# 4) t-SNE nos dados originais escalados (sem autoencoder)
tsne = TSNE(n_components=2, init="pca", perplexity=30, learning_rate="auto", random_state=SEED)
emb_tsne = tsne.fit_transform(X_scaled)

# 5) ÚNICA figura com os três plots
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
sc0 = axs[0].scatter(emb_ruim[:, 0], emb_ruim[:, 1], c=y, cmap='tab10', alpha=0.75, s=22)
axs[0].set_title("Autoencoder RUIM (linear e raso)")
axs[0].set_xlabel("z1"); axs[0].set_ylabel("z2")

sc1 = axs[1].scatter(emb_bom[:, 0], emb_bom[:, 1], c=y, cmap='tab10', alpha=0.75, s=22)
axs[1].set_title("Autoencoder MELHOR (profundo + ReLU)")
axs[1].set_xlabel("z1"); axs[1].set_ylabel("z2")

sc2 = axs[2].scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=y, cmap='tab10', alpha=0.75, s=22)
axs[2].set_title("t-SNE (dados escalados)")
axs[2].set_xlabel("dim 1"); axs[2].set_ylabel("dim 2")

# Uma única barra de cores para as classes
cbar = fig.colorbar(sc2, ax=axs, location='right', fraction=0.02, pad=0.02)
cbar.set_label('Classe')

plt.tight_layout()
plt.savefig("aann_wine_ruim_melhor_tsne.png", dpi=160)
plt.show()

print("Figura salva como: aann_wine_ruim_melhor_tsne.png")
