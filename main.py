import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# Ref: https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
# Ref: https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
# Ref: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# Ref: (To plot 100 result images) https://medium.com/the-data-science-publication/how-to-plot-mnist-digits-using-matplotlib-65a2e0cc068

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=28*10, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        # self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        # self.encConv2 = nn.Conv2d(16, 32, 5)
        # Input size = torch.Size([100, 1, 28, 28])

        # lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        # batch size, sequence length, input dimension -> Give input like this
        # 100, 28, 28 -> our dimension is 28 (one row) and we have 28 rows in a sequence.

        # Variables for LSTM
        input_dim = 28
        hidden_dim = 10
        n_layers = 1
        batch_size = 100
        seq_len = 28

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

        self.hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
        self.cell_state = torch.randn(n_layers, batch_size, hidden_dim)
        self.hidden = (self.hidden_state, self.cell_state)

        # Linear Layers
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(280, imgChannels, kernel_size = 28, stride = 1)

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
        x = x.reshape(100, 28*10)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def sampling(self, mu, log_var):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 280, 1, 1)
        x = torch.sigmoid(self.decConv1(x))
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD, BCE, KLD


def train(epoch, vae, train_loader, optimizer):
    vae.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        # Check GPU:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            data = data.cuda()

        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, log_var)

        hidden = vae.hidden
        if isinstance(hidden, tuple):
            hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            hidden = hidden.detach()
        vae.hidden = hidden

        loss.backward()
        train_loss += loss.item()
        train_kld += kld
        train_bce += bce
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    print('====> Epoch: {} Average BCE: {:.4f}'.format(epoch, train_bce / len(train_loader.dataset)))
    print('====> Epoch: {} Average KLD: {:.4f}'.format(epoch, train_kld / len(train_loader.dataset)))


def test(vae, test_loader):
    vae.eval()
    test_loss = 0

    # Check GPU:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for data, _ in test_loader:
            if device == 'cuda':
                data = data.cuda()
            recon, mu, log_var = vae(data)

            # sum up batch loss
            loss, _, _ = loss_function(recon, data, mu, log_var)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main():

    print('this is hw3')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    bs = 100
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    vae = VAE()
    if torch.cuda.is_available():
        vae.cuda()

    optimizer = optim.Adam(vae.parameters())

    # Training:
    for epoch in range(1, 4):
        train(epoch, vae, train_loader, optimizer)
        test(vae, test_loader)

    images, labels = iter(test_loader).next()
    print('Inputs shape:', images.shape)
    sample_pics = images

    # sample_pic = test_dataset[4]
    plt.imshow(sample_pics[12].reshape(28, 28), cmap="gray")
    plt.show()

    with torch.no_grad():

        if device == 'cuda':
            sample_pics = sample_pics.cuda() # Need to be shape (1,1,28,28)
        print('Pic shape:', sample_pics.shape)
        result = vae.forward(sample_pics)
        print('Got result')
        result = result[0].cpu()
        plt.imshow(result[12].reshape(28, 28), cmap="gray")
        plt.show()

    # Get randomized results:
    images, labels = iter(test_loader).next()
    print('Inputs shape:', images.shape)
    sample_pics = torch.randn(images.shape)

    # sample_pic = test_dataset[4]
    plt.imshow(sample_pics[12].reshape(28, 28), cmap="gray")
    plt.show()

    with torch.no_grad():

        if device == 'cuda':
            sample_pics = sample_pics.cuda()  # Need to be shape (1,1,28,28)
        print('Pic shape:', sample_pics.shape)
        result = vae.forward(sample_pics)
        print('Got result')
        result = result[0].cpu()
        plt.imshow(result[15].reshape(28, 28), cmap="gray")
        plt.show()

        # Plotting:
        num_row = 10
        num_col = 10  # plot images
        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
        for i in range(100):
            ax = axes[i // num_col, i % num_col]
            ax.imshow(result[i].reshape(28, 28), cmap='gray')
            # ax.set_title('Label: {}'.format(labels[i]))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()