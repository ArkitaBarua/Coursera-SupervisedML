# ### Exercise 5
# 
# Please complete the `compute_cost_reg` function below
# The starter code adds this to the cost without regularization (which you computed above in `compute_cost`) to calculate the cost with regulatization.

# UNQ_C5
def compute_cost_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar, float) Controls amount of regularization
    Returns:
      total_cost : (scalar)     cost 
    """

    m, n = X.shape
    
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b) 
    
    # You need to calculate this value
    reg_cost = 0.
    
    ### START CODE HERE ###
    for i in range(n):
        reg_cost+=w[i]*w[i]
    ### END CODE HERE ### 
    
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + (lambda_/(2*m))*reg_cost

    return total_cost

# Run the cell below to check your implementation of the `compute_cost_reg` function.

X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)

# UNIT TEST    
#compute_cost_reg_test(compute_cost_reg)

# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Regularized cost : <b></td>
#     <td> 0.6618252552483948 </td> 
#   </tr>
# </table>
