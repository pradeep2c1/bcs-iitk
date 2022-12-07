# Activation Functions
## What are activation functions, and why do we need them?
Activation functions define the output of the neurons for a set of inputs. In a broader sense, activation functions prevent linearity, i.e. if we take a linear activation function, we end up with an output, which is a linear function irrespective of whether our data is linear or not linear.
![Figure](./bcs-iitk/blog_assets/Linear-NonLinear.jpg)  
To introduce non-linearity in our dataset, we use activation functions. For peeps more interested in the technicalities, activation functions counter the problem of vanishing and exploding gradients, which arise as a result of the deeper networks where the value of derivative terms is usually less than $1$. When we multiply the derivative terms successively, the gradient becomes smaller and eventually vanishes. Otherwise, when the derivative terms are greater than $1$, their multiplication leads to gradients that tend to infinity. Activation functions counter this problem by limiting the value of the gradients. Such results can be achieved with the help of different mathematical functions used for neural network computing.
## Why does the activation function need to be non-linear?
If we have a linear activation function, we get a model which is a linear combination of inputs. Activation functions cant be linear because such a model is effectively only one layer deep. Input to networks is usually linear transformation, but real-world problems are non-linear. To make the incoming data non-linear, we use non-linear mapping called activation functions.

# Sigmoid Function
The sigmoid function is a smooth, differentiable function that introduces non-linearity to our model.
$$z = \sum_{i=1}^n w_i.x_i + w_0$$
$$\sigma(z) = {1 \over 1+e^{-z}}$$
![Graph](./bcs-iitk/blog_assets/1_OUOB_YF41M-O4GgZH_F2rw.webp)  
It outputs results between 0 and 1, and used in the output layers of the DL architectures and hence used for probability-based output, and has been applied successfully in binary classification problems, modelling logistic regression tasks, as well as other neural network domains. Derivative of the sigmoid function is given by $\sigma(z)*(1-\sigma(z))$.  
The sigmoid function is usually used in shallow networks because of problems like sharp damp gradients during backpropagation from hidden layers to the input layers. The sigmoid function saturates and kills the gradient; hence initializing the weights with values too big or too small can create a problem. Other problems faced by the sigmoid function are slow convergence and non-zero centered output thereby causing the gradient updates to propagate in different directions.  
Other functions like ReLu or tangent function were proposed to remedy some of these drawbacks.

# Hyperbolic Tangent Function $(tanh)$
The hyperbolic tangent function, known as a $tanh$ function, is a smooth and zero-centered function whose output lies between $-1$ to $1$.
$$f(z) = {e^z - e^{-z} \over e^z + e^{-z}}$$
![Graph](./bcs-iitk/blog_assets/graph-tanh(x).png.crdownload)  
Tanh function is a more preferred function compared to the sigmoid function as it gives better training performance for multi-layer neural networks. $tanh$ has a $0$ centric function; hence we can easily identify the values as either strongly positive, negative or neutral. It helps in centring the data and makes learning for the next layer much easier.  
The derivative of tanh is given by $g(z) = f'(z) = tanh^2z$; hence tanh also faces the same problem of vanishing gradient descent.  
![Graph](./bcs-iitk/blog_assets/detanh.png)  
Although both sigmoid and tanh face vanishing gradient problems, $tanh$ is $zero$ centric, and the gradients are not restricted to move in certain directions. Therefore, in practice, tanh non-linearity is always preferred to sigmoid non-linearity.  
The tanh functions have been used mostly in recurrent neural networks for natural language processing and speech recognition tasks.  

# Rectified linear activation function (ReLU)
The rectified linear activation function, or ReLU function, is a non-linear activation function. It returns $0$ for non-positive values and $“x”$ for any positive value $“x”$. The graph for this function consists of two straight lines with slope $0$ (for $x < 0$) and $1$ (for $x > 0$).
$$ReLU = max(z, 0)$$
![Graph](./bcs-iitk/blog_assets/1_Wo9l7tEb4jjY-ZGow4PzAg.png)  
Unlike the sigmoid function, its derivative is discontinuous at $0$. This creates a problem as the derivative of an activation function is crucial in updating the weights after each iteration. To solve this problem, we take the derivative to be $0$ at $x = 0$.  
The ReLU function is quite helpful in increasing the training speed of the model. This is because the derivative is always either $0$ or $1$ for any value. This saves up a lot of time in computation during backpropagation.  
A general problem with both the sigmoid and tanh functions is that they saturate. This means that large values snap to $1.0$ and small values snap to $-1$ or $0$ for tanh and sigmoid, respectively. Further, the functions are only really sensitive to changes around the mid-point of their input, such as $0.5$ for sigmoid and $0.0$ for tanh. ReLU adds more sensitivity to the weighted sum, and thus, this prevents neurons from getting saturated (i.e. when there is little or no variation in the output).  
One of the problems of ReLU is the one of "dead neurons", which occurs when the neuron gets stuck in the negative side and constantly outputs zero. Because gradient of $0$ is also $0$, it's unlikely for the neuron to ever recover. This happens when the learning rate is too high or the negative bias is quite large.
## Leaky ReLU
The Leaky ReLU activation function is just a modified form of ReLU, so to counter the disadvantages of ReLU. For the case of Leaky ReLU, it returns $x$ for positive values of $x$, and $a$ times $x$ for negative values of $x$, where $a$ is a parametric value. So, the gradient for positive inputs is $1$, and for negative inputs, it is $a$.
$$f(x) = \begin{cases}
    ax &\text{for } x < 0 \\
    x &\text{for } x \ge 0
\end{cases}$$
The advantage of leaky ReLU is the same as that of ReLU, in addition to solving the “dead neuron” response. Now, as for negative inputs, the gradient is ‘a’, and the neurons are able to recover with this minor modification.
![Graph](./bcs-iitk/blog_assets/1_FDOyQlRurCK7mWU5i0Ly_w.png)  
For choosing the value of parametric $a$, if it is very small, then its learning rate will be very low, and the model becomes time-consuming. So, by performing backpropagation, the most appropriate value of $a$ is learnt and worked accordingly.

# Softmax Function
The softmax activation function is among the most popular activation functions. It is usually placed as the last layer in the deep learning model as it is used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes.  
Softmax is an activation function that scales numbers/logits into probabilities. The output of a Softmax is a vector (say $v$) with probabilities of each possible outcome. The probabilities in vector $v$ sums to one for all possible outcomes or classes.  
Mathematically, Softmax is defined as, $$S(z)_i = {e^{z_i} \over \sum_{j = 1}^n e^{z_j}}$$
the numerator is a standard exponential function applied on $z_i$ the result is a small value (close to $0$ but never $0$) if $z_i < 0$ and a large $z_i$ is large. A normalization term $(\sum_{j=1}^n e^{z_j})$ ensures the values of output vector sums to $1$ for $i$-th class, and each of them is in the range $0$ and $1$ which makes up a valid probability distribution.
![Graph](./bcs-iitk/blog_assets/Softmax-function-image.png)  
Or more physically, it can be observed that the Softmax function calculates the probabilities distribution of the event over ‘n’ different events. In general way of saying, this function will calculate the probabilities of each target class over all possible target classes. Later the calculated probabilities will be helpful for determining the target class for the given inputs.
