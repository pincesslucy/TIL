import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch


class CelebeA_VAE(BaseModel):
    def __init__(self, latent_dim = 100):
        super(CelebeA_VAE, self).__init__()
        self.latent_dim = latent_dim
        # 64x64x3 images input
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256*4*4, self.latent_dim)  # Fully connected layer for mean
        self.fc_logvar = nn.Linear(256*4*4, self.latent_dim)  # Fully connected layer for log variance
        self.fc_decode = nn.Linear(self.latent_dim, 256*4*4)  # Fully connected layer to start decoding
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z = self.fc_decode(z)
        z = z.view(-1, 256, 4, 4)
        return self.decoder(z), mu, logvar

