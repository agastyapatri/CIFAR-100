"""
Python class to train and test any given model 
"""

from cProfile import label
from email.mime import image
import torch 
import torch.nn as nn 
import numpy as np 
from time import time

class Trainer(nn.Module):
    """
    Training Neural Networks.
        
    1. Params:

        1.1 num_epochs : The number of training iterations
        1.2 learning_rate : The rate at which the gradients converge to the global minimum.
        1.3 momentum : idk
    
    
    2. Methods:

        2.1 train(): Training any network

            :params network: Neural Nets in the form of nn.Sequential objects.
    """
    
    
    
    def __init__(self, num_epochs, learning_rate, momentum, loss_fn = None):
        super().__init__()
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        if loss_fn == "CEL":
            self.criterion = nn.CrossEntropyLoss() 
        
        elif loss_fn == "MSE":
            self.criterion = nn.MSELoss()
        
        else:
            raise Exception("Choose between CrossEntropyLoss (CEL) and Mean Squared Error(MSE)")



        
    def train_network(self,  network, dataloader, optimizer):
        
        """
        :params arch: Type of network architecture
        :params network: NN defined in Pytorch 
        :params dataloader: torch.utils.data.DataLoader object
        :params optimizer: optimizer of the weight/bias convergence.
        """
        training_loss = []
        training_accuracy = []

        if optimizer == "SGD":
            optimizer = torch.optim.SGD(network.parameters(), lr=self.learning_rate, momentum=self.momentum)

        elif optimizer == "Adam":
            optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)

        
    
        # Core Training 1: Iterating over all the Epochs
        for epoch in range(self.num_epochs):
            running_loss = 0.0

            
            # Core Training 2: Iterating over each batch 
            for idx, data in enumerate(dataloader):
                image_batch, label_batch = data

                # setting the gradients to 0 before training the batch. 
                optimizer.zero_grad()
                predictions = torch.max(network(image_batch), dim=1)[0].to(torch.float32)

                


                with torch.autograd.set_detect_anomaly(True):
                    
                    # Finding Loss
                    loss = self.criterion(predictions, label_batch.to(torch.float32))

                    # Finding Gradients (Backpropagation)
                    loss.backward(retain_graph=True)

                    # # Updating Weights and Biases
                    optimizer.step()     

                running_loss += loss.item() 

                print(running_loss)
                break 



            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Training Loss = x, Training Accuracy = y")

                
            break                 





        


class Tester(nn.Module):
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
