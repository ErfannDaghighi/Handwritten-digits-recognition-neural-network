# Handwritten Digit Recognition with PyTorch

This project implements a fully connected neural network to recognize handwritten digits from the **MNIST** dataset. It uses PyTorch for tensor operations and neural-network layers, while **backpropagation and gradient-descent parameter updates are implemented manually**, without calling `loss.backward()` or using a PyTorch optimizer.

## Requirements

- **Python:** developed with Python **3.13** on macOS (Apple Silicon).
- **Third-party Python packages:** `torch`, `numpy`, `matplotlib`.
- **Standard-library modules:** `struct`, `array`, `os.path`, `random`, `pathlib` (no separate installation required).

## MNIST dataset

The script expects the **uncompressed IDX files** in an `input/` directory alongside the script:

```text
project-root/
├── digits.py
├── input/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
└── README.md
```

Download MNIST from the [official MNIST dataset page](https://yann.lecun.com/exdb/mnist/) or a reliable mirror, decompress the four files if necessary, and make sure their filenames match those shown above. The dataset is not downloaded automatically by the script.

## Run

From the repository directory, with the virtual environment activated:

```bash
python digits.py
```

This program implements a 3-layer neural network, with an input of 784 neurons (the pixels of a 28*28 image of a handwritten digit), and an output of 10 neurons giving the probability distribution for the digit associated to the image. 

It uses the MNIST dataset of 60,000 images for training and 10,000 images for testing. I used the code from their website to read the data. 

I wrote the gradient computation function myself in order to deeply understand the formulas used by PyTorch's tools. I used the ReLU function in the first two layers and the sigmoid function in the last layer. 


## Training details

The implementation computes gradients using the chain rule and matrix/tensor operations (including matrix multiplication and outer products), then updates weights and biases manually using mini-batch gradient descent. PyTorch automatic differentiation is disabled for the model parameters.

Training hyperparameters such as the number of epochs and the learning rate are configured directly in the Python script. Results can vary with random weight initialization and training settings.

When executed, the code must run for about 10 minutes, and be updating you every 30 seconds on which epoch has been completed (out of 20 epochs). It also updates you with the current success rate. It will then give you, as an output, the graph of the cost function over the gradient descent.

The final success rate is usually about 97%. 

## Effect of noise on the neural network

We add gaussian stains of variable size and intensity to either images for training or for testing.

There are two different approaches : either we train the model on damaged images and we look at its performance to see if it can still learn. Or we train the model normally and we see if he can recognize damaged images. 
Performance seems to remain at the same level for stains of 0.5 intensity and radius 2, for both situations. 

