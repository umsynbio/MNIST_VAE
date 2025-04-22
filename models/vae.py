"""Fully‑connected VAE for 28×28 images."""
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dims=None, latent_dim: int = 20):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]

        # Encoder
        encoder_layers, prev = [], input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        decoder_layers, prev = [], latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        decoder_layers += [nn.Linear(prev, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder_layers)

    @staticmethod
    def reparameterize(mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon.view(x.size(0), 1, 28, 28), mu, logvar