### Exercise 4
# 
# Please complete the `predict` function to produce `1` or `0` predictions given a dataset and a learned parameter vector $w$ and $b$.
#   if $f(x^{(i)}) >= 0.5$, predict $y^{(i)}=1$
#   if $f(x^{(i)}) < 0.5$, predict $y^{(i)}=0$

# UNQ_C4
# GRADED FUNCTION: predict

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
    ### START CODE HERE ### 
    # Loop over each example
    for i in range(m):   
        z= 0
        # Loop over each feature
        for j in range(n): 
            z+=w[j]*X[i][j]
        z+=b
        f=sigmoid(z)
        
        # Apply the threshold
        if(f>=0.5):
            p[i] = 1
        else :
            p[i]=0
        
    ### END CODE HERE ### 
    return p

# Test your predict code
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5

tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')

# UNIT TESTS        
#predict_test(predict)

# **Expected output** 
# 
# <table>
#   <tr>
#     <td> <b>Output of predict: shape (4,),value [0. 1. 1. 1.]<b></td>
#   </tr>
# </table>

# Now let's use this to compute the accuracy on the training set


#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))


# <table>
#   <tr>
#     <td> <b>Train Accuracy (approx):<b></td>
#     <td> 92.00 </td> 
#   </tr>
# </table>

# ## 3 - Regularized Logistic Regression
# 
# In this part of the exercise, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly. 
# 
# ### 3.1 Problem Statement
# 
# Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. 
# - From these two tests, you would like to determine whether the microchips should be accepted or rejected. 
# - To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.
# 
# ### 3.2 Loading and visualizing the data
# 
# Similar to previous parts of this exercise, let's start by loading the dataset for this task and visualizing it. 
# 
# - The `load_dataset()` function shown below loads the data into variables `X_train` and `y_train`
#   - `X_train` contains the test results for the microchips from two tests
#   - `y_train` contains the results of the QA  
#       - `y_train = 1` if the microchip was accepted 
#       - `y_train = 0` if the microchip was rejected 
#   - Both `X_train` and `y_train` are numpy arrays.

# load dataset
X_train, y_train = load_data("data/ex2data2.txt")


# #### View the variables
# The code below prints the first five values of `X_train` and `y_train` and the type of the variables.

# print X_train
print("X_train:", X_train[:5])
print("Type of X_train:",type(X_train))

# print y_train
print("y_train:", y_train[:5])
print("Type of y_train:",type(y_train))

# #### Check the dimensions of your variables
# 
# Another useful way to get familiar with your data is to view its dimensions. Let's print the shape of `X_train` and `y_train` and see how many training examples we have in our dataset.


print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

# #### Visualize your data

# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()


# Figure 3 shows that our dataset cannot be separated into positive and negative examples by a straight-line through the plot. Therefore, a straight forward application of logistic regression will not perform well on this dataset since logistic regression will only be able to find a linear decision boundary.
# 

# <a name="3.3"></a>
# ### 3.3 Feature mapping
# 
# One way to fit the data better is to create more features from each data point. In the provided function `map_feature`, we will map the features into all polynomial terms of $x_1$ and $x_2$ up to the sixth power.
# 
# $$\mathrm{map\_feature}(x) = 
# \left[\begin{array}{c}
# x_1\\
# x_2\\
# x_1^2\\
# x_1 x_2\\
# x_2^2\\
# x_1^3\\
# \vdots\\
# x_1 x_2^5\\
# x_2^6\end{array}\right]$$
# 
# As a result of this mapping, our vector of two features (the scores on two QA tests) has been transformed into a 27-dimensional vector. 
# 
# - A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will be nonlinear when drawn in our 2-dimensional plot. 
# - We have provided the `map_feature` function for you in utils.py. 


print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)


print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])

