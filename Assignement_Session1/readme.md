# Assignment Session 1
## Convolution Neural network

This file contains the a simple Convolution neural network trained on MNIST dataset 

The accuracy score of network was found out to be = [0.028923396404099185, 0.9904]

## Defination

**Convolution**

The weighted sum of dot matrix multiplication of all the elements of image with the kernel.

**Filters/Kernels**

The filter/kernel extracts the particular feature when convolved with the input image. The output of the kernel is a feature which it extracts

**Epochs**

The nuber of times a network has seen/trained on  all the images in a data set. If there are 1000 image in a data set then for 1 epoc the network has seen/said to be trained on 1000 images once.

**1x1 Convolution**

It combines the feature which are linked together and reduces the number of channels.

**3x3 Convolution**

When the size of the kernel is considered as 3x3, the kernel is convolved with the input image it is said to be 3x3 convolution

**Feature Maps** 

It can also be called as a channel. It is the collection of a particular feature. Example if 'E' is a feature extracted by 'E' kernel/feature extractor then the collection of 'E' is a feature map.   

**Activation Function**

Activation functions decides whether the information that the neuron is receiving should be passed or ignored.

**Receptive Field**

Global receptive field - All the total input pixels from different channels which has undergone convolution to form one particular channel is the global receptive field of the this particular channel. 

Local receptive field - The input image pixels and the kernel is convolved to get channel, the input pixels which has undergone convolution to form this particular channel is the local receptive field of the channel. 
