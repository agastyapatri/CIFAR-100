"""
Python class to train and test any given model 
"""

from email.mime import image
import torch 
torch.manual_seed(0)

import torch.nn as nn 
import numpy as np 
from time import sleep, time
dtype = torch.float32



class Trainer(nn.Module):
    """
    """
    
    
    def __init__(self, network, num_epochs, learning_rate, momentum):
        super().__init__()

        self.network = network 
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.loss_fun = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.network.parameters(), lr = self.learning_rate, momentum=0.9)



    def train_one_epoch(self, dataloader, epoch):
        """
            Function to train a single epoch of the training cycle
        """
        self.network.train(True)
        running_loss = 0.0 
        num_batches = len(dataloader) 

        for idx, data in enumerate(dataloader):
            image_batch, label_batch = data 
            print(len(image_batch), len(label_batch))
            break 







class Tester(nn.Module):
    """
    Testing the neural network 
    """ 

    def __init__(self):
        super().__init__()
        
    def test_model(self, dataloader):

        """
        """

        images, labels = next(iter(dataloader))


        pass 




     



if __name__ == "__main__":

    sample_data = torch.randn(10, 100) 
    sample_net = nn.Sequential(
        nn.Linear(100, 8),
        nn.ReLU(),

        nn.Linear(8, 1),
        nn.LogSoftmax(dim=1)
    )

    training = Trainer(num_epochs=100, learning_rate=0.001, momentum=0.9)
    # training.train_network(network=sample_net, dataloader=, optimizer="SGD")
