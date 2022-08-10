"""
Defining the different architectures that will be used to classify the image. 
"""

import torch 
import torch.nn as nn 
import numpy as np 


class MultiLayerPerceptron(nn.Module):
    """
    Defining a Simple MLP to get baseline performance 
    
    Structure:

        1. Input Layer: Flatten > Linear Transformation > ReLU 
        2. Hidden Layer 1: Input Layer > Linear Transformation > ReLU
        3. Hidden Layer 2: Hidden Layer 1 > Linear Transformation > ReLU
        4. Output Layer: Hidden Layer 2 > Linear Transformation > LogSoftmax Transformation > Output. 
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size



    def network(self):
        mlp = nn.Sequential(
            # Input Layer
            
            nn.Flatten(),
            nn.Linear(self.input_size, self.hidden_size[0], dtype=torch.float32),
            nn.LeakyReLU(),

            # Hidden Layer 1
            nn.Linear(self.hidden_size[0], self.hidden_size[1], dtype=torch.float32),
            nn.LeakyReLU(),

            # Hidden Layer 2
            nn.Linear(self.hidden_size[1], self.hidden_size[2], dtype=torch.float32),
            nn.LeakyReLU(),

            # OutPut Layer
            nn.Linear(self.hidden_size[2], self.output_size, dtype=torch.float32),
            nn.LogSoftmax(dim=1)
        )
        return mlp 








class ConvolutionalNetwork(nn.Module):
    """
    Defining a CNN to classify the images.
    
    1. network(): Defining the CNN. 
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


    def network():
        pass 
    
    pass 



if __name__ == "__main__":
    mlptest = MultiLayerPerceptron(input_size=10, hidden_size=[8, 5, 4], output_size=1)



    







