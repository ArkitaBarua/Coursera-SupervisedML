#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression
# 
# In this exercise, you will implement logistic regression and apply it to two different datasets. 
# 
# 
# # Outline
# - [ 1 - Packages ](#1)
# - [ 2 - Logistic Regression](#2)
#   - [ 2.1 Problem Statement](#2.1)
#   - [ 2.2 Loading and visualizing the data](#2.2)
#   - [ 2.3  Sigmoid function](#2.3)
#   - [ 2.4 Cost function for logistic regression](#2.4)
#   - [ 2.5 Gradient for logistic regression](#2.5)
#   - [ 2.6 Learning parameters using gradient descent ](#2.6)
#   - [ 2.7 Plotting the decision boundary](#2.7)
#   - [ 2.8 Evaluating logistic regression](#2.8)
# - [ 3 - Regularized Logistic Regression](#3)
#   - [ 3.1 Problem Statement](#3.1)
#   - [ 3.2 Loading and visualizing the data](#3.2)
#   - [ 3.3 Feature mapping](#3.3)
#   - [ 3.4 Cost function for regularized logistic regression](#3.4)
#   - [ 3.5 Gradient for regularized logistic regression](#3.5)
#   - [ 3.6 Learning parameters using gradient descent](#3.6)
#   - [ 3.7 Plotting the decision boundary](#3.7)
#   - [ 3.8 Evaluating regularized logistic regression model](#3.8)
# 

# _**NOTE:** To prevent errors from the autograder, you are not allowed to edit or delete non-graded cells in this lab. Please also refrain from adding any new cells. 
# **Once you have passed this assignment** and want to experiment with any of the non-graded code, you may follow the instructions at the bottom of this notebook._

# <a name="1"></a>
# ## 1 - Packages 
# 
# First, let's run the cell below to import all the packages that you will need during this assignment.
# - [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
# - [matplotlib](http://matplotlib.org) is a famous library to plot graphs in Python.
# -  ``utils.py`` contains helper functions for this assignment. You do not need to modify code in this file.

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

get_ipython().run_line_magic('matplotlib', 'inline')


# <a name="2"></a>
# ## 2 - Logistic Regression
# 
# In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university.
# 
# <a name="2.1"></a>
# ### 2.1 Problem Statement
# 
# Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. 
# * You have historical data from previous applicants that you can use as a training set for logistic regression. 
# * For each training example, you have the applicant’s scores on two exams and the admissions decision. 
# * Your task is to build a classification model that estimates an applicant’s probability of admission based on the scores from those two exams. 
# 
# <a name="2.2"></a>
# ### 2.2 Loading and visualizing the data
# 
# You will start by loading the dataset for this task. 
# - The `load_dataset()` function shown below loads the data into variables `X_train` and `y_train`
#   - `X_train` contains exam scores on two exams for a student
#   - `y_train` is the admission decision 
#       - `y_train = 1` if the student was admitted 
#       - `y_train = 0` if the student was not admitted 
#   - Both `X_train` and `y_train` are numpy arrays.
# 

# In[83]:


# load dataset
X_train, y_train = load_data("data/ex2data1.txt")


# #### View the variables
# Let's get more familiar with your dataset.  
# - A good place to start is to just print out each variable and see what it contains.
# 
# The code below prints the first five values of `X_train` and the type of the variable.

# In[84]:


print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))


# Now print the first five values of `y_train`

# In[85]:


print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))


# #### Check the dimensions of your variables
# 
# Another useful way to get familiar with your data is to view its dimensions. Let's print the shape of `X_train` and `y_train` and see how many training examples we have in our dataset.

# In[86]:


print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))


# #### Visualize your data
# 
# Before starting to implement any learning algorithm, it is always good to visualize the data if possible.
# - The code below displays the data on a 2D plot (as shown below), where the axes are the two exam scores, and the positive and negative examples are shown with different markers.
# - We use a helper function in the ``utils.py`` file to generate this plot. 
# 
# <img src="images/figure 1.png" width="450" height="450">
# 
# 

# In[87]:


# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()


# Your goal is to build a logistic regression model to fit this data.
# - With this model, you can then predict if a new student will be admitted based on their scores on the two exams.

# <a name="2.3"></a>
# ### 2.3  Sigmoid function
# 
# Recall that for logistic regression, the model is represented as
# 
# $$ f_{\mathbf{w},b}(x) = g(\mathbf{w}\cdot \mathbf{x} + b)$$
# where function $g$ is the sigmoid function. The sigmoid function is defined as:
# 
# $$g(z) = \frac{1}{1+e^{-z}}$$
# 
# Let's implement the sigmoid function first, so it can be used by the rest of this assignment.
# 
# <a name='ex-01'></a>

# <a name="2.4"></a>
# ### 2.4 Cost function for logistic regression
# 
# In this section, you will implement the cost function for logistic regression.
# 
# <a name='ex-02'></a>

# <a name="2.5"></a>
# ### 2.5 Gradient for logistic regression
# 
# In this section, you will implement the gradient for logistic regression.
# 
# Recall that the gradient descent algorithm is:
# 
# $$\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & b := b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \newline       \; & w_j := w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j := 0..n-1}\newline & \rbrace\end{align*}$$
# 
# where, parameters $b$, $w_j$ are all updated simultaniously

# 
# <a name='ex-03'></a>
# 
# <a name="2.8"></a>
# ### 2.8 Evaluating logistic regression
# 
# We can evaluate the quality of the parameters we have found by seeing how well the learned model predicts on our training set. 
# 
# You will implement the `predict` function below to do this.
# 

# <a name='ex-04'></a>
# # 
# <a name="3.4"></a>
# ### 3.4 Cost function for regularized logistic regression
# 
# In this part, you will implement the cost function for regularized logistic regression.
# 
# Recall that for regularized logistic regression, the cost function is of the form
# $$J(\mathbf{w},b) = \frac{1}{m}  \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$
# 
# Compare this to the cost function without regularization (which you implemented above), which is of the form 
# 
# $$ J(\mathbf{w}.b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\right]$$
# 
# The difference is the regularization term, which is $$\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$ 
# Note that the $b$ parameter is not regularized.

# <a name='ex-05'></a>


# <a name='ex-06'></a>
