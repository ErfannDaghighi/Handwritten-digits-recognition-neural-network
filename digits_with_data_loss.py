
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import random
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test) 



input_path = Path(__file__).resolve().parent / "input"
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()




def linearise(u):
    L = [0 for _ in range(784)]
    for i in range(28):
        for j in range(28):
            L[i*28 + j] = u[i][j] / 255.0
    return torch.tensor(L, dtype=torch.float32)

#This function allows to create a 784-array from a 28*28 matrix, with values only beetwen 0 to 1 (the pixels have initial values from 0 to 255).


couche1 = nn.Linear(784, 128)
couche2 = nn.Linear(128, 64)
couche3 = nn.Linear(64, 10)

#Deactivation of automated gradient follow of the three layers

for couche in [couche1, couche2, couche3]:
    for parametre in couche.parameters():
        parametre.requires_grad_(False)

def sigmoid_list(L):
    return torch.sigmoid(L)

def sigmoid_prime_list(L):
    s = torch.sigmoid(L)
    return s * (1 - s)

def sigmoid(x):
    return torch.sigmoid(x)

def sigmoid_prime(x):
    s = torch.sigmoid(x)
    return s * (1 - s)

def ReLU_list(L):
    return torch.relu(L)

def ReLU_prime_list(L):
    return (L > 0).float()

def ReLU(x):
    if x > 0 : 
        return torch.tensor(x)
    else:
        return torch.tensor(0)

def ReLU_prime(x):
    if x > 0 : 
        return torch.tensor(1)
    else:
        return torch.tensor(0)
    

def scalar_to_array(n):
    L = [0 for _ in range (10)]
    L[n] = 1
    return L

def distance(s1, s2):
    return torch.mean((s1 - s2)**2)


# Calcul du gradiant par backpropagation manuelle

def cost_function_gradiant(image, valeur):

    y = torch.tensor(scalar_to_array(valeur))
    a_0 = image
    z_1 = couche1(a_0)
    a_1 = ReLU_list(z_1)
    z_2 = couche2(a_1)
    a_2 = ReLU_list(z_2)
    z_3 = couche3(a_2)
    a_3 = sigmoid_list(z_3)

    cost = distance(a_3, y)

    dC_dA3 = 2*(a_3 - y) 
    dC_dA2 = couche3.weight.T @ (sigmoid_prime_list(z_3) * dC_dA3)
    dC_dA1 = couche2.weight.T @ (ReLU_prime_list(z_2) * dC_dA2)

    deltaW_3 = torch.zeros(10, 64)
    deltaW_2 = torch.zeros(64, 128)
    deltaW_1 = torch.zeros(128, 784)

    deltab_3 = torch.zeros(10)
    deltab_2 = torch.zeros(64)
    deltab_1 = torch.zeros(128)

    deltab_3 = sigmoid_prime_list(z_3)*dC_dA3
    deltaW_3 = torch.outer(deltab_3, a_2)

    deltab_2 = ReLU_prime_list(z_2)*dC_dA2
    deltaW_2 = torch.outer(deltab_2, a_1)
    
    deltab_1 = ReLU_prime_list(z_1)*dC_dA1
    deltaW_1 = torch.outer(deltab_1, a_0)

    return (deltaW_3, deltaW_2, deltaW_1, deltab_3, deltab_2, deltab_1, cost)
            

# On fait 600*20 pas de gradient pour minimiser la fonction de coût (descente de gradient)

cost_over_time = []
time = []
n_epoch = 20

for epoch in range(n_epoch):    

    nu = 0.1*(0.99**epoch)

    #entraînement du modèle :
    for i in range(600):

        training_list = [(linearise(x_train[i*100+j]),y_train[i*100+j]) for j in range(100)]

        deltaW_3_tot = torch.zeros(10, 64)
        deltaW_2_tot = torch.zeros(64, 128)
        deltaW_1_tot = torch.zeros(128, 784)

        deltab_3_tot = torch.zeros(10)
        deltab_2_tot = torch.zeros(64)
        deltab_1_tot = torch.zeros(128)

        cost_mini_patch = 0

        for j in range(100):
            gradients = cost_function_gradiant(training_list[j][0],training_list[j][1])

            deltaW_3_tot += gradients[0]
            deltaW_2_tot += gradients[1]
            deltaW_1_tot += gradients[2]

            deltab_3_tot += gradients[3]
            deltab_2_tot += gradients[4]
            deltab_1_tot += gradients[5]

            cost_mini_patch += gradients[6]

        deltaW_3_tot = deltaW_3_tot / 100
        deltaW_2_tot = deltaW_2_tot / 100
        deltaW_1_tot = deltaW_1_tot / 100

        deltab_3_tot = deltab_3_tot / 100
        deltab_2_tot = deltab_2_tot / 100
        deltab_1_tot = deltab_1_tot / 100

        cost_over_time.append(cost_mini_patch / 100)
        time.append(600 * epoch + i)

        with torch.no_grad():

            couche3.weight -= nu*deltaW_3_tot
            couche2.weight -= nu*deltaW_2_tot
            couche1.weight -= nu*deltaW_1_tot

            couche3.bias -= nu*deltab_3_tot
            couche2.bias -= nu*deltab_2_tot
            couche1.bias -= nu*deltab_1_tot

    #validation du modèle
    compteur = 0

    for i in range (10000):

        a_0 = linearise(x_test[i])
        z_1 = couche1(a_0)
        a_1 = ReLU_list(z_1)
        z_2 = couche2(a_1)
        a_2 = ReLU_list(z_2)
        z_3 = couche3(a_2)
        
        y_pred = torch.argmax(z_3).item()

        if y_pred == y_test[i]:
            compteur +=1

    print(f"Epoch {epoch+1}/{n_epoch} terminée")
    print(f"Success rate of epoch {epoch+1}/{n_epoch} : {(compteur/10000)*100:.2f} %")



plt.plot(time, cost_over_time)
plt.xlabel('Numéro du pas de gradiant')
plt.ylabel('Fonction de coût')
plt.title(f"Fonction de coût au fur et à mesure de la descente de gradiant. Success rate final : {(compteur/10000)*100:.2f} %")
plt.show()


    




    
