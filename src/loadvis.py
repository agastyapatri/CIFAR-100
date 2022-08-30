"""
Python class to load and transform the data.
"""

import torch
torch.manual_seed(0)

import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt 


class Loader(nn.Module):
    """
    Class to load the CIFAR-100 data from torchvision and return a tensor 
    """

    def __init__(self, PATH, num_batches):
        super().__init__()
        self.path = PATH
        self.num_batches = num_batches

    def __str__(self):
        return f"Trainloader length = {len(self.getdata()[0])} | Testloader length = {len(self.getdata()[1])}"
        


    def labelmap(self):
        """
        Function that returns the labels of the dataset
        """
        idx = range(0, 100)

        return idx 


    def getdata(self):
        """
        Function to download and store the data into dataloaders. 
        """

        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( (0, 0,0) , (1, 1, 1))
        ])

        traindata = datasets.CIFAR100(self.path, train = True, download = True, transform = transformer)

        testdata = datasets.CIFAR100(self.path, train = False, download = True, transform = transformer)

        trainloader = DataLoader(traindata, batch_size=self.num_batches, shuffle=True)
        testloader = DataLoader(testdata,  batch_size=self.num_batches, shuffle=False)

        return trainloader, testloader

    def getanimage(self):
        """
        Function to get one image with the label 
        """
        pass 


class Visualizer:
    """
    Class to visualize stuff
    """

    def __init__(self) -> None:
        pass

    def visualize_filter(self, conv):
        """
        method to plot the activation maps of the conv layer
        """
        depth, width, height = conv.size()
        
        plt.title("Acitvation Maps of the conv layer")
        # fig, ax = plt.subplots(depth/2, depth/2)
        # for row in range(depth/2):
        #     for column in range(depth/2):
        #         ax.plot()
        
        pass 
        




if __name__ == "__main__":
    """
        Testing the class defined above.
    """        
    loader = Loader(PATH="/home/agastya123/PycharmProjects/CIFAR-100/data", num_batches = 64)
    

    





 









