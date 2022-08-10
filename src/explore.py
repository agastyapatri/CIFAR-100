"""
Python Class that enables exploration of the data. This includes statistics, actual images, etc
"""
import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt

class Explorer(nn.Module):
    """
    Looking into the CIFAR-100 dataset

    1. visualize(): Function to view the images.
  
        :param channel: the index of the channel wanted to be viewed. 0: Red, 1: Green, 2: Blue.  

        :param idx: the index of the image needed to be viewed
        
        :return: the image

    2. statistics(): Function to find the class distribution information. 

    """

    def __init__(self, loader) -> None:
        super().__init__()
        self.loader = loader 


    def visualize(self, idx, channel, show = False):
 
        dataiter = iter(self.loader)
        image, label = dataiter.next()

        channel_dict = {
            0: "Blue-Green",
            1: "Red-Blue",
            2: "Green-Red"
        }

        plt.title(f"Sample image for the class {label[idx]} with the channels: {channel_dict[channel]}")        
        plt.imshow(image[idx, channel, :, :].numpy())
        
        if show == True:
            plt.show()


    # def statistics(self):

    #     pass


        
        


if __name__ == "__main__":
    expl = Explorer(loader = None)


