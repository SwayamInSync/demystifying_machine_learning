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

In this section we'll be using Python and the formulas we derive in the previous section to create a Python class that will be able to perform Linear Regression by using Gradient Descent as optimizing algorithm to work on 3 different datasets.

>Note: All the code files can be found on Github through [this link](https://github.com/practice404/demystifying_machine_learning/tree/master/linear_regression_gradient_descent).
>
>***And it's highly recommended to follow the notebook along with this section for better understanding.***

Before we dive into writing code one important observation is to keep in mind that before using gradient descent, <u>***it's always helpful to normalize the features around its mean***.</u> The reason is that initially in dataset we can have many independent features and they can have way different values, like on average number of bedrooms can be 3-4 but the area of house feature can have way large values. Normalizing makes all the values of different features to lie on a comparable range and it also makes easier for algorithm to identify the patterns.
$$
x_{normalized} = \frac{x - \mu}{\sigma} \\
where \ \mu : mean \ and \ \sigma: standard \ deviation
$$


```python
class LinearRegression:
    def __init__(self) -> None:
        self.X = None
        self.Y = None
        self.parameters = None
        self.cost_history = []
        self.mu = None
        self.sigma = None
    
    def calculate_cost(self):
        """
        Returns the cost and gradients.
        parameters: None
        
        Returns:
            cost : Caculated loss (scalar).
            gradients: array containing the gradients w.r.t each parameter

        """

        m = self.X.shape[0]

        y_hat = np.dot(self.X, self.parameters)
        y_hat = y_hat.reshape(-1)
        error = y_hat - self.Y

        cost = np.dot(error.T, error)/(2*m) # Modified way to calculate cost

        gradients = np.zeros(self.X.shape[1])

        for i in range(self.X.shape[1]):
            gradients[i] = np.mean(error * self.X[:,i])

        return cost, gradients


    def init_parameters(self):
        """
        Initialize the parameters as array of 0s
        parameters: None
        
        Returns:None

        """
        self.parameters = np.zeros((self.X.shape[1],1))


    def feature_normalize(self, X):
        """
        Normalize the samples.
        parameters: 
            X : input/feature matrix
        
        Returns:
            X_norm : Normalized X.

        """
        X_norm = X.copy()
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)

        self.mu = mu
        self.sigma = sigma

        for n in range(X.shape[1]):
            X_norm[:,n] = (X_norm[:,n] - mu[n]) / sigma[n]
        return X_norm

    def fit(self, x, y, learning_rate=0.01, epochs=500, is_normalize=True, verbose=0):
        """
        Iterates and find the optimal parameters for input dataset
        parameters: 
            x : input/feature matrix
            y : target matrix
            learning_rate: between 0 and 1 (default is 0.01)
            epochs: number of iterations (default is 500)
            is_normalize: boolean, for normalizing features (default is True)
            verbose: iterations after to print cost
        
        Returns:
            parameters : Array of optimal value of weights.

        """
        self.X = x
        self.Y = y
        self.cost_history = []
        if self.X.ndim == 1: # adding extra dimension, if X is a 1-D array
            self.X = self.X.reshape(-1,1)
            is_normalize = False
        if is_normalize:
            self.X = self.feature_normalize(self.X)
        self.X = np.concatenate([np.ones((self.X.shape[0],1)), self.X], axis=1)
        self.init_parameters()

        for i in range(epochs):
            cost, gradients = self.calculate_cost()
            self.cost_history.append(cost)
            self.parameters -= learning_rate * gradients.reshape(-1,1)

            if verbose:
                if not (i % verbose):
                    print(f"Cost after {i} epochs: {cost}")

        return self.parameters


    def predict(self,x, is_normalize=True):
        """
        Returns the predictions after fitting.
        parameters: 
            x : input/feature matrix
        
        Returns:
            predictions : Array of predicted target values.

        """
        x = np.array(x, dtype=np.float64) # converting list to numpy array
        if x.ndim == 1:
            x = x.reshape(1,-1)
        if is_normalize:
            for n in range(x.shape[1]):
                x[:,n] = (x[:,n] - self.mu[n]) / self.sigma[n]
        x = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)
        return np.dot(x,self.parameters)
```

The class and it's methods are pretty obvious, try to go one by one and you'll understand what each method is doing and how it's connected to others.

Still I would like to put your focus on 3 main methods:

- **`calculate_cost`** : This method actually uses the formulas we derive in the previous section to calculate the cost according to certain parameters. If you carefully go through the method you may find a wierd thing that initially we mentioned cost as:
  $$
  J(\theta) = \frac{1}{2m} \sum^{m}_{i=1}\left ( X^{i}\theta - y^{i} \right )^{2}
  $$
  but in code we are calculating cost as:
  $$
  J(\theta) = \frac{1}{2m} (X\theta - Y)^{T}(X\theta - Y)
  $$
  No need to be puzzled they both are same thing, the second equation is the vectorized form of the first one. If you're aware of Linear Algebra operations you can prove to yourself that they both are same equations. We preferred the second one because oftenly vectorized operations are faster and efficient instead of using loops.

- **`fit`**: This is the method where actual magic happens. It firstly normalize the features then add an extra feature of 1s for the bias term and lastly it iterates till the epochs count, calculate the cost and gradients then update each parameter simultaneously.

  > We first normalize the features then we add an extra feature of 1s for bias term, because don't make any sense to normalize that extra feature which contain all 1s

- **`predict`**: This method first normaliz the input then uses the optimal paramteres calculated by the `fit` method to return the predicted target values.

  > Note: `predict` method uses the same &mu; and &sigma; that we calculated during the training loop from training set to normalize the input.

Great now we have our class, it's time to test it on the datasets.

### Testing on datasets

In this sub-section we'll be using Sklearn's generated dataset for linear regression to see how our Linear Regression class is performing. Let's visulaize it on graph:

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im7.png" style="zoom:50%;" />

Let's create an instance of `LinearRegression` class and fit this data on it for 500 epochs to get the optimal parameters for our hypothesis.

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im8.png" style="zoom:50%;" /> 

Okay, let's see how this hypothesis looks:

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im9.png" style="zoom:50%;" />

It fits it nicely, but plotting the cost is a great way to assure that everything is working fine, let's do that. `LinearRegression` class had a property of `cost_history` it stores the cost after each iteration, let's plot it:

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im10.png" style="zoom:50%;" />

We can see that our cost function is always decreasing and it's a good sign that our model is working pretty good.

Before moving on to next section and discuss about Regularization, I want to demostrate how we can also fit curve instead of straight line, let's see it in the next sub-section.

### Polynomial Regression

We basically going to take the generated dataset for linear regression from sklearn and apply some transformation on it to make it non-linear.

***<u>Note: For detailed code implementation I recommend you going through the notebook from [here](https://github.com/practice404/demystifying_machine_learning/blob/master/linear_regression_gradient_descent/notebook.ipynb), since for the sake of learning I'm only showing few code cells for verification</u>***

So what we did is generated the data from Sklearn's `make_regression` having 1 feature and a target column then apply the following transformation on it to make that data non-linear
$$
Y = 4(X + 1)^2 + X^5
$$
and after applying the dataset, it looks like:

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im15.png" style="zoom:50%;" />

Looks good we are able to introduce non-linearity but it'll be great if it also contains some noise samples, anyway let's start working on this non-linear dataset.

To make our linear regression to predict non-linear hypothesis we need to create more features (since we have only 1 here) from the features we already have. A good way to create more features is to perform some polynomial functions on the original features one-by-one. For thix example we are going to make 6 different features from the original one as:
$$
X1 = X \\
X2 = X^4\\
X3 = X^9\\
X4 = X^6\\
X5 = X^8\\
X6 = e^{X}
$$
We will be using these X1, X2, ..., X6 as features to make our final input/feature matirx **X_**. Now let's use this **X_** matrix to predict the optimal curve.

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im16.png" style="zoom:50%;" />

It looks great, our algorithm is abled to predict a fine non-linear boundary and it fits our training set very precisely. But there's a problem, we can see that our algorithm is performing very well on the training set but it's possible that it won't work good on the data outside the training set. This is known as **Overfitting** and it leads to the lack of generality in our hypothesis. 

We are going to address this problem using Regularization in the next section.

## Regularization

With the help of Regularization we can prevent the problem of overfitting from our algorithm. Overfitting occurs when the algorithm provides heavy parameters to some features according to the training dataset and hyperparameters. This makes those features to dominate in the overall hypothesis and lead to a nice fit in training set but not so good on the samples outside the training set.

The plan is to add the square of parameters by multiplying them with some big number (&lambda;) to the cost function because our algorithms's main motive is to decrease the cost function so in this way algorithm will end up giving the small parameters just to cancel the effect addition of parameters by multiplying with large number. So our final cost function gets modified to:
$$
J(\theta) = \frac{1}{2m} \sum^{m}_{i=1}\left ( X^{i}\theta - y^{i} \right )^{2} + \frac{\lambda}{2m}\sum^{n}_{j=1} \theta^{2}_{j}
$$
***Note: We denote the bias term as &theta;<sub>0</sub> and it's not needed to regularized the bias term that's why we are only considering only &theta;<sub>1</sub> to &theta;<sub>n</sub> parameters.***

Since our cost function is changed that's why our formulas for gradients will also be affected. The new formula for gradient will gets modified to:
$$
\frac{\partial}{\partial \theta_{j}}J(\theta) = \frac{1}{m} \sum^{m}_{i=1}\left ( X^{i}\theta - y^{i} \right )(X^{i}_{j}) + \frac{\lambda}{m}\theta_{j}
$$
&lambda; is known as regularization parameter and it should be greater than 0. Large value of &lambda; leades to underfitting and very small values lead to overfitting, so you need to pick the right one for your dataset through iterating on some sample values.

Let's implement the Regularization by modifying our `LinearRegression` class. We only need to modify the `calculate_cost` method since only this method is responsible for calculating cost and gradients both. The modified version is shown below:

```python
class LinearRegression:
    def __init__(self) -> None:
        self.X = None
        self.Y = None
        self.parameters = None
        self.cost_history = []
        self.mu = None
        self.sigma = None
    
    def calculate_cost(self, lambda_=0):
        """
        Returns the cost and gradients.
        parameters: 
            lambda_ : value of regularization parameter (default is 0)
        
        Returns:
            cost : Caculated loss (scalar).
            gradients: array containing the gradients w.r.t each parameter

        """
        m = self.X.shape[0]

        y_hat = np.dot(self.X, self.parameters)
        y_hat = y_hat.reshape(-1)
        error = y_hat - self.Y

        cost = (np.dot(error.T, error) + lambda_*np.sum((self.parameters)**2))/(2*m)

        gradients = np.zeros(self.X.shape[1])

        for i in range(self.X.shape[1]):
            gradients[i] = (np.mean(error * self.X[:,i]) + (lambda_*self.parameters[i])/m)

        return cost, gradients


    def init_parameters(self):
        """
        Initialize the parameters as array of 0s
        parameters: None
        
        Returns:None

        """
        self.parameters = np.zeros((self.X.shape[1],1))


    def feature_normalize(self):
        """
        Normalize the samples.
        parameters: 
            X : input/feature matrix
        
        Returns:
            X_norm : Normalized X.

        """
        X_norm = self.X.copy()
        mu = np.mean(self.X, axis=0)
        sigma = np.std(self.X, axis=0)

        self.mu = mu
        self.sigma = sigma

        for n in range(self.X.shape[1]):
            X_norm[:,n] = (X_norm[:,n] - mu[n]) / sigma[n]
        return X_norm

    def fit(self, x, y, learning_rate=0.01, epochs=500, lambda_=0, is_normalize=True, verbose=0):
        """
        Iterates and find the optimal parameters for input dataset
        parameters: 
            x : input/feature matrix
            y : target matrix
            learning_rate: between 0 and 1 (default is 0.01)
            epochs: number of iterations (default is 500)
            is_normalize: boolean, for normalizing features (default is True)
            verbose: iterations after to print cost
        
        Returns:
            parameters : Array of optimal value of weights.

        """
        self.X = x
        self.Y = y
        self.cost_history = []
        if self.X.ndim == 1: # adding extra dimension, if X is a 1-D array
            self.X = self.X.reshape(-1,1)
            is_normalize = False
        if is_normalize:
            self.X = self.feature_normalize()
        self.X = np.concatenate([np.ones((self.X.shape[0],1)), self.X], axis=1)
        self.init_parameters()

        for i in range(epochs):
            cost, gradients = self.calculate_cost(lambda_=lambda_)
            self.cost_history.append(cost)
            self.parameters -= learning_rate * gradients.reshape(-1,1)

            if verbose:
                if not (i % verbose):
                    print(f"Cost after {i} epochs: {cost}")

        return self.parameters


    def predict(self,x, is_normalize=True):
        """
        Returns the predictions after fitting.
        parameters: 
            x : input/feature matrix
        
        Returns:
            predictions : Array of predicted target values.

        """
        x = np.array(x, dtype=np.float64) # converting list to numpy array
        if x.ndim == 1:
            x = x.reshape(1,-1)
        if is_normalize:
            for n in range(x.shape[1]):
                x[:,n] = (x[:,n] - self.mu[n]) / self.sigma[n]
        x = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)
        return np.dot(x,self.parameters)
```

Now we have our regularized version of `LinearRegression` class. Let's address the previous problem of overfitting on polynomial regression by using a set of values for &lambda; to pick the right one.

<img src="/Users/swayam/Desktop/demystifying_machine_learning/linear_regression_gradient_descent/images/im17.png" style="zoom:50%;" />

From the plots, I think that &lambda; = 10 and &lambda; = 20 looks good. As we can see that as we increase the values of &lambda;, our algorithm start to perform even worst on the training set and leading to **Underfitting**. So it gets really important to select the right value of &lambda; for our dataset.

## Conclusion

Great work everyone, we have successfully learnt and implemented Linear Regression using Gradient Descent. There are a few things that we need to keep in our mind that this optimizing algorithm requires more hyperparameters than the Normal equaltion that we learnt in previous article but irrespective of that, gradient descent works efficiently on the larger dataset covering the drawback of Normal equation method.

In the next article we'll be learning our first supervised classification algorithm known as **Logistic Regression** and going to understand how to Regularization prevents the overfitting there. 

I hope you have learnt something new, for more updates on upcoming articles get connected with me through [Twitter](https://twitter.com/_s_w_a_y_a_m_) and stay tuned for more. Till then enjoy your day and keep learning.
