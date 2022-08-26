"""
Python file that binds together all the other modules, and runst the classification procedure from start to finish. 
"""


# Global Imports
import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt

# Custom Imports 1
from src.traintest import Trainer, Tester
from src.loadvis import Loader
# Custom Imports 2
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

    # 1.  Loading the data 
    trainloader, testloader = Loader(PATH=path, num_batches=128).getdata()

    # 2. Defining the Network 
    model = ConvolutionalNetwork(input_channels=32, output_size=100)
     


    