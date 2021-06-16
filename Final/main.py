import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from model import VAE

# Ref: https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
# Ref: https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
# Ref: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# Ref: (To plot 100 result images) https://medium.com/the-data-science-publication/how-to-plot-mnist-digits-using-matplotlib-65a2e0cc068

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


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

    return train_loss / len(train_loader.dataset)


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
    train_loss_to_plot = []
    for epoch in range(1, 6):
        train_loss = train(epoch, vae, train_loader, optimizer)
        train_loss_to_plot.append(train_loss)
        test(vae, test_loader)

    # Saving the trained model:
    PATH = './vae_model.pth'
    torch.save(vae.state_dict(), PATH)

    # show loss curve
    plt.plot(train_loss_to_plot)
    plt.show()

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