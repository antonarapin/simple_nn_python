# Simple Neural Network in Python Using Numpy

## Prerequisites

All you need is a working version of Python and Numpy. 

To install Python, visit [Python's official website](https://www.python.org/downloads/).

To install numpy:

```
pip install numpy
```

## Some Theoretical Basics

### Neural Nets

Simple neural network described in this project is nothing more than a fully connected graph. We will look at the example with two hidden layers, so the layers of the network are: input layer, two hidden layers, output layer. 

Here is the layout:

![alt text](https://github.com/antonarapin/simple_nn_python/blob/project_description/images/net_layout.png "Fully connected graph")

(Thanks to the author [this repo](https://github.com/martisak/dotnets) for typing up a quick and pretty visualization tool)

#### Forward Pass

Each edge has a weight, and inside each node there is some sort of activation function that is applied to the data that is coming in. Here is how everything happens:

1. Inputs are weighted in each node of the first hidden layer

2. Weighted inputs are summed in each node of the first hidden layer

3. Activation function is applied to the sum

4. The resulting value from each node in the first hidden layer is propagated forward to the second layer, now treating the outputs of the first hidden layer as inputs into second hidden layer

In mathematical terms, we can express what happens in each node in the following way:

![alt text](https://github.com/antonarapin/simple_nn_python/blob/project_description/images/inside_node_sum.png "What happens in each node")

This expression is exactly the ouput value of each node in the network. Here, *v<sub>i</sub>* is the *i<sup>th</sup>* input into the *j<sup>th</sup>* node, multiplied by the weight *w<sub>ij</sub>*, which is diplayed as the edge connecting input node *i* to the node *j*. The mysterious φ symbol is the activation function, and in this example we will use ReLU(Rectifier Linear Unit) and Sigmoid activation functions. Both of these functions are pretty straight-forward and common, so I will just show the formulas.

Sigmoid function: ![alt text](https://github.com/antonarapin/simple_nn_python/blob/project_description/images/sigmoid.png "Sigmoid") 

ReLU function: ![alt text](https://github.com/antonarapin/simple_nn_python/blob/project_description/images/relu.png "ReLU")

The operation described above happens in each node in each layer, going from left to right from the first hidden layer to the output layer. Eventually, the outputs of the nodes in the ouput layer are the outputs of our network. And that is pretty much it for the forward pass, very straight-forward (yes, you have been PUNished)! 

<sub>I know, people regard puns as something simple and often react to them like "euhh, yeah", but I think puns are awesome! They are the examples of inline creativity, embellishing the expression with their beautiful brefity and ingenuity.</sub>

### Linear Algebra

## Implementation

At this point, neural network might seem like an atrocious thing to implement, and it for sure might be so, especially if you do it in c (yes, a new project is coming). But fear not! Neural networks with Python and Numpy are very laconic and cute. 


