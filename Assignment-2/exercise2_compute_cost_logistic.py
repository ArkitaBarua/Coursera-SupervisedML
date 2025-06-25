# ### Exercise 2
# 
# Please complete the `compute_cost` function.
# Note:
# * As you are doing this, remember that the variables `X_train` and `y_train` are not scalar values but matrices of shape ($m, n$) and ($𝑚$,1) respectively, where  $𝑛$ is the number of features and $𝑚$ is the number of training examples.
# * You can use the sigmoid function that you implemented above for this part.
# 

# UNQ_C2
# GRADED FUNCTION: compute_cost
import math
def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """

    m, n = X.shape
    
    ### START CODE HERE ###
    
    total_cost=0
    for i in range(m):
        z=0
        for j in range(n):
            z+=w[j]*X[i][j]
        z+=b
        f=sigmoid(z) #computed in exercise-1
        total_cost=total_cost-y[𝑖]*math.log(f)-(1-y[𝑖])*math.log(1-f)
    ### END CODE HERE ### 

    return (1/m)*total_cost
  
# Run the cells below to check your implementation of the `compute_cost` function with two different initializations of the parameters $w$ and $b$

m, n = X_train.shape

# Compute and display cost with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Cost at initial w and b (zeros)<b></td>
#     <td> 0.693 </td> 
#   </tr>
# </table>


# Compute and display cost with non-zero w and b
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at test w and b (non-zeros): {:.3f}'.format(cost))


# UNIT TESTS
#compute_cost_test(compute_cost)


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Cost at test w and b (non-zeros):<b></td>
#     <td> 0.218 </td> 
#   </tr>
# </table>
