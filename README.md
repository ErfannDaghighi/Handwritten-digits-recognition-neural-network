# Handwritten-digits-recognition-neural-network
Implementation of a neural network in Python/PyTorch but with handwritten function to compute the gradient, in order to have a more intuitive understanding of the gradient descent of a learning neural network.

This program implements a 3-layer neural network, with an input of 784 neurons (the pixels of a 28*28 image of a handwritten digit), and an output of 10 neurons giving the probability distribution for the digit associated to the image. 

It uses the MNIST dataset of 60,000 images for training and 10,000 images for testing. See https://www.kaggle.com/datasets/hojjatk/mnist-dataset for more information. I used the code from this website to read the data. 

I wrote the gradient computation function myself in order to deeply understand the formulas used by PyTorch's tools. I used the ReLU function in the first two layers and the sigmoid function in the last layer. 

When executed, the code must run for about 10 minutes, and be updating you every 30 seconds on which epoch has been completed (out of 20 epochs). It also updates you with the current success rate. It will then give you, as an output, the graph of the cost function over the gradient descent.

The final success rate is usually about 97%. 


