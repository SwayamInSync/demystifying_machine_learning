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
Cost \ function: \\\\ J(\theta) = \frac{1}{2m} \sum^{m}_{i=1}\left ( \hat{y}_{i} - y_{i} \right )^{2} \\
J(\theta) = \frac{1}{2m} \sum^{m}_{i=1}\left ( X_{i}\theta - y_{i} \right )^{2} 
\\
where \ X_{i} \ is \ the \ i^{th} \ sample \ in \ dataset
$$
It's important to notice that our cost function ***J(&theta;)*** is depend upon the parameters &theta; because our **y**(target) and **X** are fixed, the only varying quantity are the parameters &theta; and it makes sense because that's how gradient descent will help us in finding the appropriate parameters for minium of the cost function.

## Mathematics of Gradient Descent

