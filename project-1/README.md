# Project-1: Cats and Dogs Classification using Logistic Regression

<img src="./article_images/head.jpg" style="zoom:50%;" />

## Overview

Welcome to the 4<sup>th</sup> article of [**Demystifying Machine Learning** ](https://swayam-blog.hashnode.dev/series/demystifying-ml) series. In this article, we'll be going to make our first project of ***prediction of cats or dogs from their respective images*** using Logistic Regression. We are going to use [scikit-learn](https://scikit-learn.org/stable/) library for this project but I already covered [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) in great depth in the 3<sup>rd</sup> article of this series. 

>  *** The main focus of this sample project is towards data collection and data preprocessing rather than training a Logistic Regression model. The reason is that I showed how to set different hyperparameters in order to achieve satisfying results in [3<sup>rd</sup> article](https://swayam-blog.hashnode.dev/logistic-regression) of this series and often times whenever working on a project 80% of the time one gonna spent is on data preprocessing.***



The project is going to be a [**Jupyter Notebook**](https://github.com/practice404/demystifying_machine_learning/blob/master/project-1/project_1.ipynb). You can grab that notebook from [here](https://github.com/practice404/demystifying_machine_learning/blob/master/project-1/project_1.ipynb). Everything you need from code to explanation is already provided in that [notebook](https://github.com/practice404/demystifying_machine_learning/blob/master/project-1/project_1.ipynb). This article will tell you few points about the dataset and some techniques we are going to use.

## About Dataset

The dataset we are going to use comes from [Kaggle](http://kaggle.com). You can view that dataset on Kaggle from [here](https://www.kaggle.com/c/dogs-vs-cats). This dataset only contains images of cats and dogs separated as train and test already for model training.

In the [notebook](https://github.com/practice404/demystifying_machine_learning/blob/master/project-1/project_1.ipynb) we'll be training our model from images of cats and dogs inside `train` folder and validate our model using images inside `test` folder. 

The dataset in total contains 11250 images of cats and 11250 images of dogs in train folder. Similarly, it contains 1250 images of cats and 1250 images of dogs in test folder. There is no labels file so we had to create it by ourselves to let our model understand which image is of a cat and which is of a dog.

## How to open?

As mentioned earlier this project is basically a Jupyter notebook and you can get that notebook from Github via this [link](https://github.com/practice404/demystifying_machine_learning/blob/master/project-1/project_1.ipynb). You don't need to download the dataset manually because inside the notebook you'll find a function that will download the dataset for you automatically and save it in your current working directory then another function will extract that dataset into a folder.

Now you can either upload this notebook on [Google Colab](http://colab.research.google.com) or any other cloud service and run it there or download it on your system and create a virtual environment then run it after installing all the required dependencies. Both ways it's totally gonna work fine without any issue.

Using cloud service can be easier because you don't have to manage virtual environments or install required dependencies but you also need to keep in mind that they provide limited RAM storage so keep a keen eye on your memory management and if it's getting out of your provided range then change the number of samples in training set inside [notebook](https://github.com/practice404/demystifying_machine_learning/blob/master/project-1/project_1.ipynb).

> ***Note: All the code sample is done in a notebook with a proper explanation of each cell***

## Conclusion

That's it for this supplement article for our first project. Now head to the [repository for this project](https://github.com/practice404/demystifying_machine_learning/tree/master/project-1) and grab the notebook. If you just want to take a look at the notebook and it's not opening on Github then click on this [link](https://nbviewer.org/github/practice404/demystifying_machine_learning/blob/master/project-1/project_1.ipynb) to explore the entire notebook on [nbviewer](https://nbviewer.org).

I hope you have learnt something new, for more updates on upcoming articles get connected with me through [Twitter](https://twitter.com/_s_w_a_y_a_m_) and stay tuned for more. 

<center><b>Wish you all a Happy New Year...!!</b></center>
