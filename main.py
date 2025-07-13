import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # Mean of q(z|x)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # Log-variance of q(z|x)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # std = sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std  # z = μ + σ * ε
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))  # Output in [0,1]
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar):
    # Reconstruction loss (BCE)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence between N(mu, sigma) and N(0, 1)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_div

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(x.size(0), -1).to(device)  # Flatten to [B, 784]

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss = vae_loss(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader.dataset):.4f}")


model.eval()
with torch.no_grad():
    x, _ = next(iter(train_loader))
    x = x.view(x.size(0), -1).to(device)
    x_recon, _, _ = model(x)

    x = x.view(-1, 1, 28, 28).cpu()
    x_recon = x_recon.view(-1, 1, 28, 28).cpu()

    # Show original and reconstructed images
    n = 8
    plt.figure(figsize=(16, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i][0], cmap='gray')
        plt.axis('off')

        # Reconstruction
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_recon[i][0], cmap='gray')
        plt.axis('off')
    plt.show()
