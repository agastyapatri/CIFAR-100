"""
Python file that binds together all the other modules, and runst the classification procedure from start to finish. 
"""


# Global Imports
import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt

# Custom Imports 1
from src.load import Tensor
from src.explore import Explorer
from src.traintest import Trainer, Tester

# Custom Imports 2
from src.models import MultiLayerPerceptron
from src.models import ConvolutionalNetwork




if __name__ == "__main__":

    """
    PROCEDURE OF CLASSIFICATION: 
        1 a. Loading the data
        1 b. Visualizing the data

        2. Defining the network
        3. Training the model 
        4. Getting Dumb Baselines
        5. Hyperparameter tuning
        6. Testing 

    """


    path ="/home/agastya123/PycharmProjects/CIFAR-100/data"

    # 1 a. 
    trainloader, testloader = Tensor(PATH=path, num_batches=64).getdata()
    
    # 1 b. 
    img = Explorer(trainloader).visualize(idx = 0, channel = 0, show=False)

    # 2. 
    mlp = MultiLayerPerceptron(input_size=3072, hidden_size=[2048, 1024, 512, 256], output_size=100)
    mlp_net = mlp.network()


    # 3.
    train = Trainer(num_epochs=1000, learning_rate=1e-6, momentum=0.9, loss_fn="MSE")
    train.train_network(network=mlp_net, dataloader=trainloader, optimizer="SGD")

    # 4. 

    # 5. 

    # 6. 
    # test = Tester()




    