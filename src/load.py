"""
Python class to load and transform the data.
"""

# from distutils.command.config import config
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import scipy
import os


class Tensor(nn.Module):
    """
    Class to load the CIFAR-100 data from torchvision and return a tensor 
    """

    def __init__(self, PATH, num_batches):
        super().__init__()
        self.path = PATH
        self.num_batches = num_batches

    def __str__(self):
        trainloader, testloader = self.getdata()

        trainshape, testshape = len(list(enumerate(trainloader))), len(list(enumerate(testloader)))                   

        return f"Training Dataset length = {trainshape}, Testing Dataset length = {testshape}"



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
        testloader = DataLoader(testdata,  batch_size=self.num_batches, shuffle=True)


        return trainloader, testloader

    def to_tensor(self):
        """
        Function to convert / view the dataloaders as usable tensors
        """    
        trainloader, testloader = self.getdata()
        
    

if __name__ == "__main__":
    """
        Testing the class defined above.
    """        
    test = Tensor(PATH="/home/agastya123/PycharmProjects/CIFAR-100/data", num_batches = 64)
    traindata, testdata = test.getdata()
    





 









