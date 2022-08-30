# **CIFAR-100 Image Classification** 
_This project aims at creating an image classification model that is trained on the CIFAR-100 dataset_
created by [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)

_Refer to my [Blog Post] for a complete documentation of my procedure of training this 
classifier_

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
Getting rudimentary baselines is always a good idea. Obviously, the minimum accuracy threshold corresponds to the model simply guessing the class an image belongs to (1% in this case, as there are 100 classes. This figure is not very useful, however). To this end, I created a simple MLP with 3 Hidden Layers and LeakyReLU activation. 
The accuracy achieved by this network will be the new baseline performance I have to beat.

A bit of an issue I faced was regarding PyTorch's handling of datatypes. _Everything in this project is of ``` torch.float32  ``` datatype._

**Note: Training for one batch**
The woes that are dogged are those of many variables. To alleviate some of this confusion, I trained the network on just one batch, for all the training epochs. Referring to Karpathy's advice, overfitting on just one batch verifies that a lowest possible loss can be achieved. An added benefit is a quicker testing cycle, as well as observing a perceptible convergence. 
https://towardsdatascience.com/designing-your-neural-networks-a5e4617027ed
**Initial Parameters Used** are as follows: 

| **Hyperparameter**     |   **Configuration** |
|------------------------|---------------------|
| Hidden Layers          | 3                   |
| Learning Rate          |     0.0001          |
| Training Epochs        |     1000            |
| Batch Size             |       64            |
| Optimizer              | Adam                |
| Loss Function          | MeanSquaredError    |

With this configuration, the training loss oscillates every ~150 epochs. Diagnosing this problem reveals that the learning rate chosen was far too high. Moving to a Gradient-Descent based algorithm might also help. Standby.

**Initial Baseline Performance for one Batch:** Training loss = 4.2935225
    



### _**2.2 Creating a (hopefully better) Convolutional Neural Network (CNN)**_

Now, the meat of this project. Creating a Convolutional Neural Network will help in detecting the features in images. 


**Tentative Struture of the CNN:**



| **Hyperparameter**     |   **Configuration** |
|------------------------|---------------------|
| CONV Layers          |             3       |
|FC Layers               |          3          |
| Learning Rate          |     1e-3          |
| Training Epochs        |     100          |
| Batch Size             |       64            |
| Optimizer              | Adam                |
| Loss Function          | MeanSquaredError    |




## **3. Experimentation**

_Baseline Training Loss after the first training cycle of 100 epochs = 4.59. This section is about experiments done to improve performance._

1. **Reducing batch size, Increasing learning rate:**
    Batch size from 128 to 64, and Learning rate from 1e-5 to 1e-3 makes the network converge faster. 

        * Training Loss at the end of 100 epochs is : 0.015348
  
        * Average Testing Loss for 157 batches is : 6.233311. 

    Clearly, the network has overfit massively to the training data. The testing loss is actually _more_ than it was when the network was completely untrained. Nevertheless, performing well on the training data is not a bad start. My initial plan was to increase the depth of the network to improve training performance, as every online treatment of this dataset uses a relatively more complex architecture in the context of transfer learning. Karpathy's advice is to pick a large enough model that overfits on the training data and then regularize. Who am I to disagree? 


2. **Redudcing the learning rate:** Reducing the learning rate from 1e-3 to 1e-4 led to:
   
    * `Training Loss at the end of 100 epochs is : 2.13376`
    
    * `Average Testing Loss for 157 batches is : 3.06784`

    Next, Finding an intermediate learning rate to balance the results


3. **Regularization:** Adding dropout, weight decay etc.



  






## **4. Results and Discussion** 
_Write about the results obtained. Should be of the form below:_

| **Model Architecture** | **Metric Obtained** |
|------------------------|---------------------|
| Model 1                | Result 1            |
| Model 2                | Result 2            |




## **5. References**
_A quasi-exhaustive list of resources that guided this enterprise._
1. [Microsoft Documentation on Training Image Classifiers](https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model)

2. [PyTorch Documentation](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)

3. [Batch Normalization on Wikipedia](https://en.wikipedia.org/wiki/Batch_normalization)

4. [Andrej Karpathy's Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)








## **Notes (Not to be included)**

_A CNN is a class of neural networks defined as multilayered NNs designed to detect complex features in data. Below are a list of definintions_

1. **The Convolutional Layer:** The Convolution layer is a main layer of a CNN which helps us detect features in images. Each of the layers has a number of channels to detect specific features in images, and a number of kernels to define the size of the detected feature. 
   In this case, each image is a tensor with dimensions (3 x 32 x 32) 


2. **Kernel:** The kernel defines the size of the image / detected feature by applying a filter to the image in question. The most general case would be the kernel being the size of the image. _In this case, the kernel size is 32, since each image is (32x32)_

3. **Batch Normalization:** Defined as a layer in the NN that normalizes the input recieved from a previous layer. This is supposed to make the training of the NN faster and more stable.

4. **Max Pooling:** An operation that calculates the maximum value of a patch of pixels, and uses it to create a downsampled feature map. It is usually used after a Convolutional Layer. Max Pooling is done in part ot help overfitting by providing an abstract form of representation. It also reduces the computational cost by reducing the number of parameters that need to be learned. 
    _MaxPool layer will help us to ensure that the location of an object in an image will not affect the ability of the NN to detect specific features._

5. **Regularization:** 
   1. Getting More Data
   2. Data Augmentation 
   3. Creative Augmentation
   4. Pretrain 
   5. Stick with Supervised learning
   6. smaller input dimensionality 
   7. smaller model size
   8. decrease batch size
   9. Add Dropout 
   10. Weight Decay 
   11. **Early Stopping** Stop Training based on masured validation loss to catch the model as it is about to overfit. 
   12. **Try a larger model**, after early stopping 




