# Linear Regression using Gradient Descent

<img src="https://images.unsplash.com/photo-1543286386-2e659306cd6c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80" style="zoom:50%;" />



## Overview

This is the second article of **Demystifying Machine Learning** series, fankly it is basically the ***sequel*** of our previous article where we explained [**Linear Regression using Normal equation**](https://swayam-blog.hashnode.dev/linear-regression-using-normal-equation). In this article we'll be exploring another optimizing algorithm known as **Gradient Descent**, how it works, what is a cost function, mathematics behined gradient descent, Python implementation, Regularization and some extra topics like polynomial regression and using regularized polynomial regression.

## How Gradient Descent works (Intuition)

Gradient descent is basically an iterative optimizing algorithm i.e. we can use it to find the minimum of a differential function. Intuitively we can think of a situation where  you're standing somewhere on a mountain and you want to go to the foot of that mountain as fast as possible. Since we're on a random position on the mountain and we need to descent fast so, one way is to move along the steepest direction while taking small steps *(taking large steps towards steepest direction may get you injured)* and we'll see later that taking large steps is also not good for algorithm too. 

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im1.png" style="zoom:50%;" />

Now that's something similar we are going to do for optimizing our hypothesis to give least error. As we learnt in the [previous article](https://swayam-blog.hashnode.dev/linear-regression-using-normal-equation) that we need to find the optimal parameters &theta;, that helps us to calculate the best possible hyperplane to fit the data. In that article we used normal equation to directly find those parameters but here that's not gonna happen. 

We are randomly going to pick the parameters and then calculate the cost function that will tell us how much error those random parameters are giving, then we use gradient descent to find the minimum of that cost function and optimizing those random parameters into the optimal ones.

## Cost Function

A cost function is basically a continuous and differentiable function that tells how good an algorithm is performing by returning the amount of error as output. The lesser the error, the better the algorithm is doing that's why we randomly generate the parameters and then keep changing them in order to reach to the minimum of that cost function.

Now let's define the cost function for Linear regression. First we need to think that in linear regression how we can calculate the error. ***Mathamtically error is the difference of original value and the calculated value.*** Luckily we can use this definition here. Our original values is the **target** matrix itself and the calculated values are the predictions from our hypothesis. 

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im2.jpg"/>

We simply subtract the original target value from the predicted target value and take the square of them as the error for single sample. Ultimately we need to find the **squared error** for all the samples in dataset and take their **mean** as our final cost for certain hypothesis. Squaring the difference helps in avoiding the condtion when the negative and positive error nullify each other in the final hypothesis's cost.

This error function also known as *** Mean Square Error (MSE).***

So mathematically let's say we have ***m*** number of samples in dataset then:
$$
Cost \ function: \\\\ J(\theta) = \frac{1}{2m} \sum^{m}_{i=1}\left ( \hat{y}^{i} - y^{i} \right )^{2} \\
J(\theta) = \frac{1}{2m} \sum^{m}_{i=1}\left ( X^{i}\theta - y^{i} \right )^{2} 
\\
where \ X^{i} \ is \ the \ i^{th} \ sample \ in \ dataset
$$
It's important to notice that our cost function ***J(&theta;)*** is depend upon the parameters &theta; because our **y**(target) and **X** are fixed, the only varying quantity are the parameters &theta; and it makes sense because that's how gradient descent will help us in finding the appropriate parameters for minium of the cost function.

## Mathematics of Gradient Descent

> Time to talk Calculus.

Before diving into the algorithm let's first talk about what is a Gradient? 

***Gradient of a differentiable function is a vector field whose value at certain point is the vector whose components are the partial derivatives of that function at that same point.*** Alight so many big words let's break them down and try to understand what it really is?

Mathematically suppose you have a function ***f(x,y,z)*** then the gradient at some point will be the vector whose components are goint to be the partial derivatives of ***f(x,y,z)*** w.r.t to **x,y** and **z** at that point.
$$
f(x,y,z)\\
f'_{x} = \frac{\partial}{\partial \ x} f(x,y,z) \\
f'_{y} = \frac{\partial}{\partial \ y} f(x,y,z) \\
f'_{z} = \frac{\partial}{\partial \ z} f(x,y,z) \\ \\
Gradient \ vector: \\
\overrightarrow{\bigtriangledown } f =  \begin{bmatrix}
f'_{x} \\
f'_{y} \\
f'_{z}
\end{bmatrix}
$$

> **Property**:  ***At a certain point, the gradient vector of that point always points towards the direction of greatest increase of that function***. ***Since we need to go in the direction of greatest decrease that's why we follow the direction of negative of the gradient vector.***
>
> ***Gradient vector is always perpendicular to the contour lines of the graph of a function***

Let's visualize the gradient concept using graphs. 
$$
f(x,y) = x^{2} + y^{2} \\ \\
\overrightarrow{\bigtriangledown } f =  \begin{bmatrix}
2x \\
2y 
\end{bmatrix}
$$
If we plot the above graph, it'll look something like this:

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im3.png" style="zoom:50%;" />

If you're aware of vector calculus, then you probably know that Contour plots are very useful for working with 3D curves. Contour plot is basically a 2D graph that is the sliced version of 3D plot along the z-axis at regular intervals, so if we graph the Contour plot of the above function then it'll look something like:

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im4.png" style="zoom:50%;" />

Now this graph makes it really clear that gradient always points in the direction of greatest increase of the function, as we can see that the black arrows represent the direction of gradient and the red arrow represent the direction where we need to move in our cost function to reach the minimum.

Great now we know that in order to reach to the mimum we need to move in the opposite direction of the gradient that is in the **-&nabla;f(&theta;)** direction and keep updating our initial random parameters accordingly.
$$
\theta _{j} = \theta _{j} - \alpha * \frac{\partial}{\partial \ \theta_{j}} J(\theta) \quad for \ all \  \theta _{s} \ simultaneously
$$

> - &theta; is the matirx of all parameters &theta;<sub>s</sub> 
> - &theta;<sub>j</sub> is the parameter for j<sup>th</sup> feature
> - J(&theta;) is the cost function
> - &alpha; is learning rate

Everything seems obvious instead of this symbol &alpha; . It's known as learning rate, remember we discussed that we need to take small steps, &alpha; makes sure that our algorithm should take small steps for reaching to the minimum. Learning rate is always less than 1.

But what is we keep large learning rate ?

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im5.jpg" style="zoom:50%;" />

As we see in above figure that our cost function will not able to reach to minimum if we take large learning rates and it result in increment of loss instead of decreasing it as represented below.

![](/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im6.png)

## Applying Gradient descent to cost function

In this section we'll be deriving the formulas for gradients so that we can directly use those formulas in Python implementaion. Since we already have our cost function as:
$$
J(\theta) = \frac{1}{2m} \sum^{m}_{i=1}\left ( X^{i}\theta - y^{i} \right )^{2}
$$
expanding X<sup>i</sup> into individual ***n*** features as [X<sup>i</sup><sub>1</sub>, X<sup>i</sup><sub>2</sub>, X<sup>i</sup><sub>3</sub>, ....., X<sup>i</sup><sub>n</sub> ] then:
$$
J(\theta) = \frac{1}{2m} \sum^{m}_{i=1}\left ( (X^{i}_{1}\theta_{1} + X^{i}_{2}\theta_{2}+ X^{i}_{3}\theta_{3}+ .... +X^{i}_{n}\theta_{n}) - y^{i} \right )^{2}
$$
This form will be easier to understand the calculation of gradients, let's compute them for each &theta;<sub>j</sub>.
$$
\frac{\partial}{\partial \theta_{1}}J(\theta) = \frac{1}{m} \sum^{m}_{i=1}\left ( (X^{i}_{1}\theta_{1} + X^{i}_{2}\theta_{2}+ X^{i}_{3}\theta_{3}+ .... +X^{i}_{n}\theta_{n}) - y^{i} \right )(X^{i}_{1})
\\
\frac{\partial}{\partial \theta_{2}}J(\theta) = \frac{1}{m} \sum^{m}_{i=1}\left ( (X^{i}_{1}\theta_{1} + X^{i}_{2}\theta_{2}+ X^{i}_{3}\theta_{3}+ .... +X^{i}_{n}\theta_{n}) - y^{i} \right )(X^{i}_{2})
\\
\frac{\partial}{\partial \theta_{3}}J(\theta) = \frac{1}{m} \sum^{m}_{i=1}\left ( (X^{i}_{1}\theta_{1} + X^{i}_{2}\theta_{2}+ X^{i}_{3}\theta_{3}+ .... +X^{i}_{n}\theta_{n}) - y^{i} \right )(X^{i}_{3})
\\
.\\
.\\
.\\
\frac{\partial}{\partial \theta_{n}}J(\theta) = \frac{1}{m} \sum^{m}_{i=1}\left ( (X^{i}_{1}\theta_{1} + X^{i}_{2}\theta_{2}+ X^{i}_{3}\theta_{3}+ .... +X^{i}_{n}\theta_{n}) - y^{i} \right )(X^{i}_{n})
$$


so basically we can write the partial derivative of cost function w.r.t to any &theta;<sub>j</sub> as :
$$
\frac{\partial}{\partial \theta_{j}}J(\theta) = \frac{1}{m} \sum^{m}_{i=1}\left ( X^{i}\theta - y^{i} \right )(X^{i}_{j})
$$
Now we can loop over each &theta;<sub>j</sub> from 1 to ***n*** and update them as :
$$
\theta _{j} = \theta _{j} - \alpha * \frac{\partial}{\partial \ \theta_{j}} J(\theta)
$$
That's great, now we have all the tools we need let's jump straight into the code and implement this algorithm in Python.

## Python Implementation

