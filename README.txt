This is the README file.
---------------------------------------------------------------------------------------
model.py
This file includes the VAE class, which includes the encoder and the decoder,
the sampling method and the forward function.
Train function is not included in this file.
Parameters regarding to network (number of layers and dimensions) can be changed
from this file.
---------------------------------------------------------------------------------------
main.py
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
------------------------------------------------------------------------------------
generator.py
When this file is running:
Model is loaded from the same directory, name of the file can be changed.
Randomized vectors are created using torch.randn to feed the decoder.
Results from the decoder are visualized in a grid.