# **CIFAR-100 Image Classification** 
_This project aims at creating an image classification model that is trained on the CIFAR-100 dataset_
created by [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)

## **1. About the dataset**
The CIFAR-100 dataset consists of 60000 32x32
colour images in 100 classes, with 600 images 
per class. There are 500 training images and 
100 testing images per class. There are 50000 
training images and 10000 test images. 
The 100 classes are grouped into 20 
superclasses. There are two labels per 
image - fine label (actual class) and coarse label
(superclass).

_Summary taken verbatim from [huggingface](https://huggingface.co/datasets/cifar100)_


## **2. Modus Operandi**

_Kudos to [Andrej Karpathy's Post](https://karpathy.github.io/2019/04/25/recipe/) on 
training neural networks_  

### _**2.1 Creating a simple Multi Layer Perceptron**_
Getting rudimentary baselines is always a good idea. Obviously, the minimum accuracy threshold corresponds to the model simply guessing the class an image belongs to (1% in this case, as there are 100 classes. This figure is not very useful, however). To this end, I created a simple MLP with 2 Hidden Layers and LeakyReLU activation. 
The accuracy achieved by this network will be the new baseline performance I have to beat.

A bit of an issue I faced was regarding PyTorch's handling of datatypes. Everything in this project is of ``` torch.float32  ``` datatype.

### _**2.2 Creating a (hopefully better) Convolutional Neural Network**_



## **3. Results and Discussion** 
_Write about the results obtained. Should be in the form below:_

| **Model Architecture** | **Metric Obtained** |
|------------------------|---------------------|
| Model 1                | Result 1            |
| Model 2                | Result 2            |

## **4. References**
_Cite every resource that guided this enterprise._

## 5. Structure of this Repository 
_Talk about the layout of the repo. What each directory relates to and so forth._




