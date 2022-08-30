"""
Python class to train and test any given model 
"""


import torch 
torch.manual_seed(0)

import torch.nn as nn 
import numpy as np 
import time 
dtype = torch.float32



class TrainTest(nn.Module):
    
    def __init__(self, network, num_epochs, learning_rate, momentum):
        super().__init__()

        self.network = network 
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.loss_fun = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.network.parameters(), 
        lr = self.learning_rate, momentum=0.9)



    def train_one_epoch(self, dataloader, epoch):
        """
            Function to train a single epoch of the training cycle
        """
        self.network.train(True)
        running_loss = 0.0 
        num_batches = len(dataloader) 
        

        for idx, data in enumerate(dataloader):
            image_batch, label_batch = data 
            self.optim.zero_grad()
            output = self.network(image_batch) 
            loss = self.loss_fun(output, label_batch)
            
            # backprop 
            loss.backward()
            self.optim.step()

            running_loss += loss.item()
            
            # reporting after one epoch 
            if (idx + 1)%num_batches == 0:
                # calculating average loss
                avg_epoch_loss = running_loss / num_batches
                print(f"Training Epoch: {epoch +1}, Average training loss = {avg_epoch_loss}")
        return self.network, avg_epoch_loss


    def train_all_epochs(self, dataloader):
        """
        Function to train the network for all the epochs
        """
        training_loss = []
        start = time.time()

        # training for all the epochs
        for e in range(self.num_epochs):
            trained_network, avg_epoch_loss = self.train_one_epoch(dataloader=dataloader, epoch=e)
            training_loss.append(avg_epoch_loss)


        end = time.time()
        print(f"\nTraining Done. Time Taken to Train the network = {round((end - start)/60)} minutes.\n")
        print(f"Training Loss at the end of {self.num_epochs} epochs is : {round(training_loss[-1], 6)}")

        torch.save(self.network.state_dict(), "/home/agastya123/PycharmProjects/CIFAR-100/figures-results/cifarclassifier.pth")

        
        return trained_network, training_loss 


    def testmodel(self, trained_net, loader):
        """
        Testing a model 
        1. load network 
        2. load testing data
        3. find testing loss for all the batches for one epoch. Copy the train_one_epoch method.
        """
        trained_net.eval()
        running_loss = 0.0
        num_batches = len(loader)
        
        for idx, data in enumerate(loader):
            image_batch, label_batch = data 

            output = trained_net(image_batch)
            loss = self.loss_fun(output, label_batch)
            running_loss += loss.item()

        avg_testing_loss = running_loss / num_batches
        print(f"Average Testing Loss for {num_batches} batches is : {round(avg_testing_loss, 6)}")




    
if __name__ == "__main__":

    sample_data = torch.randn(10, 100) 
    sample_net = nn.Sequential(
        nn.Linear(100, 8),
        nn.ReLU(),

        nn.Linear(8, 1),
        nn.LogSoftmax(dim=1)
    )

    training = TrainTest(num_epochs=100, learning_rate=0.001, momentum=0.9)
    # training.train_network(network=sample_net, dataloader=, optimizer="SGD")
