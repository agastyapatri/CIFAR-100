"""-----------------------------------------------------------------------------------
Defining the different architectures that will be used to classify the image. 
-----------------------------------------------------------------------------------"""

import torch 
torch.manual_seed(0)
dtype = torch.float32

import torch.nn as nn 
import numpy as np 



class ConvolutionalNetwork(nn.Module):
    """
    Defining a CNN to classify the images.
    :params input_channels: the number of channels in the input image
    :params output_size: size of the output linear layer.
    """
    
    def __init__(self, input_channels, output_size):
        super().__init__()
        self.in_ch = input_channels
        self.output_size = output_size

    def __str__(self):
        return f"\n Convolutional: {self.CONV()}\nFully Connected: {self.FC()}"

    def CONV(self):
        """
        Defining the convolutional part of the network
        """
        conv = nn.Sequential(
            # Input Layer. IN_WIDTH = 32, OUT_WIDTH = 
            nn.Conv2d(in_channels=self.in_ch, out_channels=12, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=12),
            


            # Layer 2. Width  = (32-5) + 1 = 28  
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=24),
            

            # Layer 3. Width = (28-5) + 1 = 24
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),

            # Pooling Layer
            nn.MaxPool2d(kernel_size=3)
            
        )
        return conv 


    def FC(self):
        """
        Defining the Fully Connected Part of the network
        """
        fc = nn.Sequential(
            nn.Linear(in_features=1152, out_features=576),
            nn.ReLU(),
            
            nn.Linear(in_features=576, out_features=288),
            nn.ReLU(),

            nn.Linear(in_features=288, out_features=144),
            nn.ReLU(),

            nn.Linear(in_features=144, out_features=100),
            nn.ReLU()
        )
        return fc 
         


    def network(self):
        CONV = self.CONV()
        FC = self.FC()

        cnn = nn.Sequential(
            CONV,
            nn.Flatten(),
            FC,
            nn.LogSoftmax(dim=1)
            )

        return cnn
    




if __name__ == "__main__":
    # sample data of one batch of images.
    testdata = torch.randn(128, 3, 32, 32)
    
    
    model = ConvolutionalNetwork(input_channels=3, output_size=100)
    

    CNN = model.network()
    print(CNN(testdata))


    







