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


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Working directly from the saved model:
    PATH = './model.pth'
    vae = VAE()
    if torch.cuda.is_available():
        vae.cuda()
    vae.load_state_dict(torch.load(PATH))

    # Get randomized results, feed to decoder:
    sample_pics = torch.randn((100, 256))
    with torch.no_grad():

        if device == 'cuda':
            sample_pics = sample_pics.cuda()  # Need to be shape (1,1,28,28)
        result = vae.decoder(sample_pics)
        result = result.cpu()

        # Plotting:
        num_row = 10
        num_col = 10  # plot images
        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
        for i in range(100):
            ax = axes[i // num_col, i % num_col]
            ax.imshow(result[i].reshape(28, 28), cmap='gray')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()