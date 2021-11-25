# Linear Regression using Normal Equation

<img src="/Users/swayam/Desktop/linear_regression/images/head.jpeg" style="zoom:50%;" />

## Overview

Linear Regression is the first algorithm in the **Demystifying Machine Learning** series and in this article we'll be discussing about Linear Regression using Normal equation. This article covers what is Linear Regression, how it works, maths behind normal equation method, fixing some edge cases, handling overfitting and code implementation.

## What is Linear Regression ?

Linear Regression in simple terms is fitting the best possible linear hypothesis *(a line or a hyperplane)* on the data having a linear relationship so that we can predict the new unknown data point with least possible error. It's not necessary to have linear relationship in data but having such can lead to approximately close predictions. For a reference take a look on the below representation.

<img src="/Users/swayam/Desktop/linear_regression/images/intro_linear_reg.png" style="zoom:50%;" />

In the above picture only 1 feature *(along x-axis)* and the target *(along y-axis)* is displayed just for the sake of sinmplicity and we can see that the red line is fitting the data very nicely covering most of the variance. 

One thing is to be noted that we call it Linear Regression but it's not always fitting a line, we call it hypothesis or hyperplane. If we have N-Dimensional data *(data having N number of features)* then we can fit a hyperplane of atmost N-Dimensions.



## Mathematics behind the scenes

Let's take a very simpler problem and dataset to derive and mimic the algorithm we are going to use in Linear Regression. 

Assume we have a dataset in which we have only 1 feature say ***x*** and target as ***y*** such that **` x = [1,2,3] `** and **` y = [1,2,2] `** and we are going to fit the best possible line on this dataset.

<img src="/Users/swayam/Desktop/linear_regression/images/plot1.png" style="zoom:50%;" />

In the above plot, we can see that the feature is displayed along *x-axis* and target is along *y-axis* and the blue line is the best possible hypothesis through the points. **Now all we need to understand is how to come up with this best possible hypothesis that is the above blue line in our case**.

The equation of any line is in the format of **`y = wx + b`** where **w** is the slope and **b** is the intercept. In machine learning lingo we call **w** as **weight** and **b** as **bias**. For above line it came out to be **w = 0.5** and **b = 0.667 ** *(Don't worry! we'll see how).*

Now ultimately we can say that somehow we need to calculate the **weight** and **bias** term / terms *(since 'x' is already known to us)* for hypothesis and then we could use them to get line's equation.

### Linear Algebra

We are going to use some Linear Algebra concepts for finding the right **weight** and **bias** term for our data. 

Let say the weight and bias term are **w** and **b** respectively. So we can write each data point (x, y) as : 
$$
w + b = 1 & for &(1,1)\\
2w + b = 2 & for &(2,2) \\
3w + b = 2 & for &(3,2) \\
$$
we can write the above equations as a system of equations using matrices as **X&theta; = Y** where **X** is input / feature matrix, **&theta;** is matrix for unknowns and **Y** is the target matrix as: 
$$
X\Theta = Y \\
\left(\begin{array}{cc} 
1 & 1 \\
2 & 1 \\
3 & 1 \\
\end{array}\right)
\left(\begin{array}{cc} 
w \\ 
b
\end{array}\right)
=
\left(\begin{array}{cc} 
1 \\
2 \\
2 \\
\end{array}\right)
$$
Great, Now all we need is to solve this system of equations and get the **w** and **b** terms.

Wait there's a problem. We can't solve the above system of equations because target matrix **Y** does not lie in the column space of input matrix **X**. In simple terms if we see the previous graph again then we can notice that our data points are not collinear i.e. they don't lie on the line so and that's why we can't find the **w** and **b** for the above system of equations.

And if we think for a moment then it sounds right because in Linear Regression we fit a hypothesis to predict the target for some input with least possible error. We do not intend to predict the exact target.

So what we can do here ? We can't solve the above system of equations because **Y** is not in the column space of **X**. So instead   we can project the **Y** in onto the column space of **X**. It exactly equivalent to projecting one vector onto another.

<img src="/Users/swayam/Desktop/linear_regression/images/Projection_and_rejection.png" style="zoom:50%;" />

In the above representation, **a** and **b** are two vectors and the **a<sub>1</sub>** is the projection of vector **a** onto **b**. With this we can see that now we have the component of vector **a** that lies in the vector space of **b**. 

We can achieve the component of **Y** that lies in the column space of **X** by doing inner product *(also known as dot product)*. 

> The inner product of two vectors **a** and **b** can be found by calculating **a<sup>T</sup>b** 

Now we can re-write our system of equations as:
$$
X \Theta = Y \\
$$

<center>multiplying both sides by X<sup>T</sup>.</center>

$$
X^{T}X\Theta = X^{T}Y \\ 
$$

<center>Assuming (X<sup>T</sup>X) to be invertible</center>

$$
\Theta = (X^{T}X)^{-1}X^{T}Y
$$

The above equation is known as **Normal equation**. Now we have a formula to find our matrix &theta;, let's use it and calculate the **w** and **b**. 
$$
X^{T}X = \begin{bmatrix}
 1&1 \\ 
 2&1\\ 
 3&1
\end{bmatrix}
\begin{bmatrix}
1 & 2 & 3\\ 
1 & 1 & 1
\end{bmatrix}
 = 
\begin{bmatrix}
 14&6 \\ 
6 & 3
\end{bmatrix}
$$

$$
X^{T}Y = \begin{bmatrix}
1 &2  &3 \\ 
1 & 1 & 1
\end{bmatrix}

\begin{bmatrix}
1\\ 
2\\ 
2\\
\end{bmatrix}

= 

\begin{bmatrix}
11\\ 
5
\end{bmatrix}
$$

$$
\Theta = (X^{T}X)^{-1}X^{T}Y \\\\
\Theta = 
\begin{bmatrix}
 14&6 \\ 
6 & 3
\end{bmatrix} ^{-1}
\begin{bmatrix}
11\\ 
5
\end{bmatrix} \\ \\
\Theta = 
\begin{bmatrix}
0.5 & -1.0\\ 
-1.0 & 2.33
\end{bmatrix}
\begin{bmatrix}
11\\ 
5
\end{bmatrix}\\ \\
\Theta = 
\begin{bmatrix}
0.5\\ 
0.6667
\end{bmatrix}\\ \\
\begin{bmatrix}
w\\ 
b
\end{bmatrix} = 
\begin{bmatrix}
0.5\\ 
0.6667
\end{bmatrix}
$$

from the above last equation we have our **w** = **0.5** and **b** = **2/3** *(0.6667)* and we can check from the equation of blue line that our **w** and **b** are exactly correct. That's how we can get the weights and bias terms for our perfect hypothesis using the ***Normal equation***.

## Python Implementation

Now it's time to get our hand dirty with code implementation and making our algorithm work for real datasets.
