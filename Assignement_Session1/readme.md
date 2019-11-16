# Assignment Session 1
## DNN accuracy score

[0.028923396404099185, 0.9904]

## Defination

**Convolution**

The weighted sum of dot matrix multiplication of all the elements of image with the kernel.

**Filters/Kernels**

The filter/kernel extracts the particular feature when convolved with the input image. The output of the kernel is a feature which it extracts

**Epochs**

The nuber of times a network has seen all the images in a data set. If there are 1000 image in a data set then for 1 epoc the network has seen/trained 1000 images once.

**1x1 Convolution**

It combines the feature which are linked together

**3x3 Convolution**

When the size of the kernel is considered as 3x3

**Feature Maps** 

It can also be called as a channel. It is the collection of a particular feature. Example if 'E' is a feature extracted by 'E' kernel/feature extractor then the collection of 'E' is a feature map.   

**Activation Function**

Activation functions decides whether the information that the neuron is receiving should be passed or ignored.

**Receptive Field**

Global receptive field- The pixels which have undergone convolution to get the particular channel as output is global receptive field of this particular channel
Local receptive field - The input image and the kernel is convolved to get channel, the pixels on the input which has undergone convolution to form this particular channel is the local receptive field of the channel. 
