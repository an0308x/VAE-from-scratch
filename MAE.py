import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_ch=1, embed_dim=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MAE(nn.Module):
    def __init__(self, img_size=28, patch_size=7, embed_dim=64, encoder_depth=2, decoder_dim=32, decoder_depth=1, mask_ratio=0.5):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.mask_ratio = mask_ratio

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Encoder
        self.encoder_blocks = nn.Sequential(*[TransformerBlock(embed_dim) for _ in range(encoder_depth)])

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.decoder_blocks = nn.Sequential(*[TransformerBlock(decoder_dim) for _ in range(decoder_depth)])
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size)  # Predict pixels

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, imgs):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed

        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Encode
        x_encoded = self.encoder_blocks(x_masked)

        # Decode
        x_dec = self.decoder_embed(x_encoded)

        # Pad masked tokens
        B, L, D = x_dec.shape
        decoder_tokens = torch.zeros(B, self.num_patches, D, device=x.device)
        len_keep = x_encoded.shape[1]
        decoder_tokens.scatter_(1, ids_restore.unsqueeze(-1).expand(-1, -1, D), x_dec)

        x_decoded = self.decoder_blocks(decoder_tokens)
        pred = self.decoder_pred(x_decoded)  # (B, N, patch_size²)
        return pred, mask

def mae_loss(pred, imgs, patch_size=7, mask=None):
    B, _, H, W = imgs.shape
    target = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    target = target.contiguous().view(B, 1, -1, patch_size, patch_size)
    target = target.view(B, -1, patch_size * patch_size)

    loss = (pred - target) ** 2
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)
    return loss.mean()

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        pred, mask = model(imgs)
        loss = mae_loss(pred, imgs, patch_size=7, mask=mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)[:8]
    pred, mask = model(imgs)

    patch_size = 7
    B, _, H, W = imgs.shape
    N = (H // patch_size) * (W // patch_size)

    target = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    target = target.contiguous().view(B, 1, -1, patch_size, patch_size).squeeze(1)

    pred_img = pred.view(B, N, patch_size, patch_size)

    # Reconstruct images
    def reconstruct(patches):
        rows = torch.chunk(patches, chunks=H // patch_size, dim=1)
        return torch.cat([torch.cat(torch.chunk(r.squeeze(1), chunks=W // patch_size, dim=0), dim=-1) for r in rows], dim=-2)

    imgs_orig = imgs.cpu().numpy()
    imgs_recon = [reconstruct(p).cpu().numpy() for p in pred_img]
    imgs_masked = [reconstruct(t * (1 - m.unsqueeze(-1).unsqueeze(-1))).cpu().numpy()
                   for t, m in zip(target, mask)]

    # Plot
    for i in range(8):
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(imgs_orig[i, 0], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(imgs_masked[i][0], cmap='gray')
        plt.title("Masked")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(imgs_recon[i][0], cmap='gray')
        plt.title("Reconstruction")
        plt.axis('off')
        plt.show()
