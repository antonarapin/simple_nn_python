# Simple Neural Network in Python Using Numpy

---

## Prerequisites

All you need is a working version of Python and Numpy. 

To install Python, visit [Python's official website](https://www.python.org/downloads/).

To install numpy:

```
pip install numpy
```

---

## Some Theoretical Basics

---

### Neural Nets

Simple neural network described in this project is nothing more than a fully connected graph. We will look at the example with two hidden layers, so the layers of the network are: input layer, two hidden layers, output layer. 

Here is the layout:

![alt text](https://github.com/antonarapin/simple_nn_python/blob/project_description/images/network_layout.png "Fully connected graph")

(Thanks to the author [this repo](https://github.com/martisak/dotnets) for typing up a quick and pretty visualization tool)

---

#### Forward Path

Each edge has a weight, and inside each node there is some sort of activation function that is applied to the data that is coming in. Here is how everything happens:

1. Inputs are weighted in each node of the first hidden layer

2. Weighted inputs are summed in each node of the first hidden layer

3. Activation function is applied to the sum

4. The resulting value from each node in the first hidden layer is propagated forward to the second layer, now treating the outputs of the first hidden layer as inputs into second hidden layer

In mathematical terms, we can express what happens in each node in the following way:

![alt text](https://github.com/antonarapin/simple_nn_python/blob/project_description/images/inside_node_sum.png "What happens in each node")

This expression is exactly the ouput value of each node in the network. Here, *v<sub>i</sub>* is the *i<sup>th</sup>* input into the *j<sup>th</sup>* node, multiplied by the weight *w<sub>ij</sub>*, which is diplayed as the edge connecting input node *i* to the node *j*. The mysterious φ symbol is the activation function, and in this example we will use ReLU(Rectifier Linear Unit) and Sigmoid activation functions. Both of these functions are pretty straight-forward and common, so I will just show the formulas.

---

Sigmoid function: 

![alt text](https://github.com/antonarapin/simple_nn_python/blob/project_description/images/sigmoid.png "Sigmoid") 

ReLU function: 

![alt text](https://github.com/antonarapin/simple_nn_python/blob/project_description/images/relu.png "ReLU")
---

The operation described above happens in each node in each layer, going from left to right from the first hidden layer to the output layer. Eventually, the outputs of the nodes in the ouput layer are the outputs of our network. And that is pretty much it for the forward path, very straight-forward (yes, you have been PUNished)! 

<sub>I know, people regard puns as something simple and often react to them like "euhh, yeah", but I think puns are awesome! They are the examples of inline creativity, embellishing the expression with their beautiful brevity and ingenuity.</sub>

---

#### Backward Path

Now we need to measure the error, figure out what weights contributed to the error and to what extent and of course fix those bad boys. 

In this example, we will calculate the difference between the target output and actual network output and call it an error. There are different ways to calculate the error, like squared error and such, but we will go with the easier example here. 

Now we need to determine how much each weight contributed to the error. In order to achieve this, we will start look at the weights that connect the last hidden layer with the ouput layer, since they are the closest to the error we have right now. But how do we calculate the contribution of a given weight to the error? The answer is the partial derivative of the error relative to a given weight. Hence, we want to find the value of *∂Error/∂w<sub>ij</sub>*, and the easiest way to find it is to simply decompose the partial derivative using chain rule as shown below.

![alt text](https://github.com/antonarapin/simple_nn_python/blob/project_description/images/err_deriv.png "Partial derivative of Error with respect to a specific weight")

Okay, let's first state what those variables mean. The first one is *out<sub>j</sub>*, which is the output of the node that takes in the value to which our weight of interest is applied, so, literally, *out<sub>j</sub>* = *φ(Σw<sub>ij</sub>* * *v<sub>i</sub>)*. Now, *lin<sub>j</sub>* can be viewed as a linear part of *out<sub>j</sub>*, in othr words (symbols, actually) *lin<sub>j</sub>* = *Σw<sub>ij</sub>* * *v<sub>i</sub>*.

Now we will spell out what each of the terms of the scarry chain rule equation above means. Notice that since we are using a simple difference between the target value and output as error value, we can get away with assuming *∂Error/∂out<sub>j</sub>* is simply equal to the error. The second term *∂out<sub>j</sub>/∂lin<sub>j</sub>* is simply equal to the derivative of the activation function, whether it is ReLU or Sigmoid. And finally the last term is *∂lin<sub>j</sub>/∂w<sub>ij</sub>*, which, is simply the value that *w<sub>ij</sub>* is being applied to, and which we denote as *v<sub>i</sub>*. 

Notice that if we start our calculation from the output layer, we can go back and use precomputed errors for each node to get the partial contributions of weights to the left of that node (as it appears on the graph). Once we determine how much each weight contributed to the overall error, we can simply "fix" that weight according to the amount of its contribution to the overall error. Also, we can control how much the weight should be fixed according to its error contribution by specifying the *learning rate*. For example, if *∂Error/∂w<sub>ij</sub>* = 0.1 and *leaning rate* = 0.5, then we update our old weight by *learning rate* * *∂Error/∂w<sub>ij</sub>* = 0.5 * 0.1, so the new value of our weight would be *w<sub>ij</sub>* = *w<sub>ij</sub>* + *learning rate* * *∂Error/∂w<sub>ij</sub>*.

Believe it or not, this is it, we are just going through the network in a backward manner starting from the output layer and "fixing" every weight on our way relative to its error contribution. And once we do it many times, the weights will eventually converge to outputting the right thing (assuming are training experiences are not to complicated for network to tackle)!

---

## Implementation

At this point, neural network might seem like an atrocious thing to implement, and it for sure might be so, especially if you do it in C (oh yes I did it, and I'll push it uphere sometime). But fear not! Neural networks with Python and Numpy are very laconic and cute. 


