import torch
import torch.nn as nn
import torch.nn.functional as F

# Ref: https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
# Ref: https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
# Ref: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# Ref: (To plot 100 result images) https://medium.com/the-data-science-publication/how-to-plot-mnist-digits-using-matplotlib-65a2e0cc068

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=28*64, zDim=256):
        super(VAE, self).__init__()

        # Input size = torch.Size([100, 1, 28, 28])
        # lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        # batch size, sequence length, input dimension -> Give input like this
        # 100, 28, 28 -> our dimension is 28 (one row) and we have 28 rows in a sequence.

        # Variables for LSTM
        input_dim = 28
        hidden_dim = 64
        n_layers = 3
        batch_size = 100
        seq_len = 28

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
        self.cell_state = torch.randn(n_layers, batch_size, hidden_dim)
        self.hidden = (self.hidden_state, self.cell_state)

        # Linear Layers
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Convolutional layers for decoder
        self.decConv1 = nn.ConvTranspose2d(256, 128, 5, padding=2, stride=1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.decConv2 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.decConv3 = nn.ConvTranspose2d(64, 16, 3, padding=1, stride=1)
        self.conv3_bn = nn.BatchNorm2d(16)
        self.decConv4 = nn.ConvTranspose2d(16, imgChannels, 28, padding=0, stride=1)

    def encoder(self, x):
        dimension = x.shape[0]
        x = x.view(dimension, 28, 28)

        hidden = self.hidden

        # Check GPU:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            if isinstance(hidden, tuple):
                hidden = (hidden[0].cuda(), hidden[1].cuda())
            else:
                hidden = hidden.cuda()

        x, hidden = self.lstm1(x, hidden)
        self.hidden = hidden

        # Arrange for linear
        x = x.reshape(100, 28*64)
        mu = self.encFC1(x)
        log_var = self.encFC2(x)
        return mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, x):
        x = x.view(-1, 256, 1, 1)
        x = self.decConv1(x)
        x = F.relu(x)
        x = self.conv2_bn(self.decConv2(x))
        x = F.relu(x)
        x = self.decConv3(x)
        x = F.relu(x)
        x = torch.sigmoid(self.decConv4(x))
        return x

    def forward(self, x):
        # encoder -> sampling -> decoder
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var

