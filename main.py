"""
Python file that binds together all the other modules, and runst the classification procedure from start to finish. 
"""


# Global Imports
import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt

# Custom Imports 1
from src.traintest import TrainTest
from src.loadvis import Loader, Visualizer
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
    trainloader, testloader = Loader(PATH=path, num_batches=32).getdata()

    # 2. Defining the Network 
    model = ConvolutionalNetwork(input_channels=3, output_size=100)
    CNN = model.network()
    
    

    # 3. Training the Network 
    traintest = TrainTest(network=CNN, num_epochs=100, learning_rate=2.5e-4, momentum=0.9)
    trained_net, training_loss = traintest.train_all_epochs(dataloader=trainloader)


    # # 4. Testing the trained network on hitherto unseen data.
    # CNN.load_state_dict(torch.load("/home/agastya123/PycharmProjects/CIFAR-100/figures-results/cifarclassifier.pth"))
    traintest.testmodel(trained_net=CNN, loader=testloader)
     

    


    

    