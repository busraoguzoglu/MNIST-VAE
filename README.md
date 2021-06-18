# MNIST-VAE

Implementation of VAE, different versions are as follows:

Linear: Encoder -> Linear | Decoder -> Linear (Ref: https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb)

CNN: Encoder -> CNN | Decoder -> CNN (Ref: https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71)

LSTM-CNN: Encoder -> LSTM | Decoder -> CNN (the main version, in the files main.py model.py and generator.py)

Implemented for the third project of CMPE 597 Deep Learning Course of Bogazici University.

-----------------------------------------------------------------------------------------------------------------
model.py: 
This file includes the VAE class, which includes the encoder and the decoder,
the sampling method and the forward function.
Train function is not included in this file.
Parameters regarding to network (number of layers and dimensions) can be changed
from this file.

-----------------------------------------------------------------------------------------------------------------
main.py: 
When this file is running:
Datasets are loaded, they are downloaded if they do not exist in the file 'mnist_data'.
Network is defined (from model.py)
Training function is called and training is done.
Testing function is called and test is done after the training.
When the training finishes, three curves are plotted:
First curve shows the change in total loss.
Second curve shows the change in KLD.
Third curve shows the change in BCE.
After the training finishes, trained model is saved as model.pth file.
Training function is called in main function, number of epochs can be changed from there.
Can be used with CUDA if available.

-----------------------------------------------------------------------------------------------------------------
generator.py: 
When this file is running:
Model is loaded from the same directory, name of the file can be changed.
Randomized vectors are created using torch.randn to feed the decoder.
Results from the decoder are visualized in a grid.
