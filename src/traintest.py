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


    def getlabel(self, logs):
        """
        Function to get the labels from softmax/logsoftmax outputs for an
        input batch 
        """
        label_idx = torch.argmax(logs, dim=1)
        return label_idx  

    
    def accuracy(self, outputs, ground_truth_batch):
        """
        function to find the accuracy of predictions  
        """
        num_correct = 0
        num_wrong = 0


        ground_truth_batch
        prediction_batch = self.getlabel(logs=outputs)
        boolean_array = prediction_batch == ground_truth_batch
        boolean_array = boolean_array.numpy()

        for val in boolean_array:
            if val == True:
                num_correct += 1
            else:
                num_wrong += 1

        accuracy = (num_correct/(num_correct + num_wrong))*100
        return accuracy
    

    def train_one_epoch(self, dataloader, epoch):
        """
            Function to train a single epoch of the training cycle
        """
        self.network.train(True)
        running_loss = 0.0 
        running_acc = 0.0 
        num_batches = len(dataloader) 
        

        for idx, data in enumerate(dataloader):
            image_batch, label_batch = data 
            self.optim.zero_grad()
            output = self.network(image_batch) 
            loss = self.loss_fun(output, label_batch)
            acc = self.accuracy(outputs=output, ground_truth_batch=label_batch)

            # backprop 
            loss.backward()
            self.optim.step()

            running_loss += loss.item()
            running_acc += acc 
            
            # reporting after one epoch 
            if (idx + 1)%num_batches == 0:
                # calculating average loss
                avg_epoch_loss = running_loss / num_batches
                avg_epoch_acc = running_acc / num_batches
                print(f"Training Epoch {epoch +1}: Average Loss = {round(avg_epoch_loss, 5)}. Average Accuracy = {round(avg_epoch_acc, 4)}%")

        return self.network, avg_epoch_loss, avg_epoch_acc


    def train_all_epochs(self, dataloader):
        """
        Function to train the network for all the epochs
        """
        training_loss = []
        training_acc = []
        start = time.time()

        # training for all the epochs
        for e in range(self.num_epochs):
            trained_network, avg_epoch_loss, avg_epoch_acc = self.train_one_epoch(dataloader=dataloader, epoch=e)
            training_loss.append(avg_epoch_loss)
            training_acc.append(avg_epoch_acc)

        end = time.time()

        # Reporting Results 
        print(f"\nTraining Done. Time Taken to Train the network = {round((end - start)/60)} minutes.\n")
        print(f"Training Loss at the end of {self.num_epochs} epochs is : {round(training_loss[-1], 6)}")
        print(f"Training Accuracy at the end of {self.num_epochs} epochs is : {round(training_acc[-1], 4)}%")

        torch.save(self.network.state_dict(), "/home/agastya123/PycharmProjects/CIFAR-100/figures-results/cifarclassifier.pth")

        
        return trained_network, training_loss, training_acc


    def testmodel(self, trained_net, loader):
        """
        Testing a model 
        """
        trained_net.eval()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = len(loader)

        
        for idx, data in enumerate(loader):
            image_batch, label_batch = data 

            output = trained_net(image_batch)
            loss = self.loss_fun(output, label_batch)
            acc = self.accuracy(output, label_batch)
            running_loss += loss.item()
            running_acc += acc 

        avg_testing_loss = running_loss / num_batches
        avg_testing_acc = (running_acc / num_batches)
        print(f"\nAverage Testing Loss for {num_batches} batches is : {round(avg_testing_loss, 6)}\nAverage Testing Accuracy is {round(avg_testing_acc,4)}%")

    

        



        










    
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
