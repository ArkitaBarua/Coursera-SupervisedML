### Exercise 3
# 
# Please complete the `compute_gradient` function 


# UNQ_C3
# GRADED FUNCTION: compute_gradient
def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ### 
    for i in range(m):
        z = 0
        for j in range(n): 
            z+=w[j]*X[i][j]
        z+=b
        f=sigmoid(z) #from exercise-1
        
        dj_db += f-y[i]
        
        for j in range(n):
            dj_dw[j] +=(f-y[i])*X[i][j] 
    ### END CODE HERE ###
    
    return (1/m)*dj_db, (1/m)*dj_dw

# Run the cells below to check your implementation of the `compute_gradient` function with two different initializations of the parameters $w$ and $b$

# Compute and display gradient with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>dj_db at initial w and b (zeros)<b></td>
#     <td> -0.1 </td> 
#   </tr>
#   <tr>
#     <td> <b>dj_dw at initial w and b (zeros):<b></td>
#     <td> [-12.00921658929115, -11.262842205513591] </td> 
#   </tr>
# </table>

# Compute and display cost and gradient with non-zero w and b
test_w = np.array([ 0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)

print('dj_db at test w and b:', dj_db)
print('dj_dw at test w and b:', dj_dw.tolist())

# UNIT TESTS    
#compute_gradient_test(compute_gradient)


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>dj_db at test w and b (non-zeros)<b></td>
#     <td> -0.5999999999991071 </td> 
#   </tr>
#   <tr>
#     <td> <b>dj_dw at test w and b (non-zeros):<b></td>
#     <td>  [-44.8313536178737957, -44.37384124953978] </td> 
#   </tr>
# </table>

# <a name="2.6"></a>
# ### 2.6 Learning parameters using gradient descent 

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value 
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant
      
    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing



np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)


# <details>
# <summary>
#     <b>Expected Output: Cost     0.30, (Click to see details):</b>
# </summary>
# 
#     # With the following settings
#     np.random.seed(1)
#     initial_w = 0.01 * (np.random.rand(2) - 0.5)
#     initial_b = -8
#     iterations = 10000
#     alpha = 0.001
#     #
# 
# ```
# Iteration    0: Cost     0.96   
# Iteration 1000: Cost     0.31   
# Iteration 2000: Cost     0.30   
# Iteration 3000: Cost     0.30   
# Iteration 4000: Cost     0.30   
# Iteration 5000: Cost     0.30   
# Iteration 6000: Cost     0.30   
# Iteration 7000: Cost     0.30   
# Iteration 8000: Cost     0.30   
# Iteration 9000: Cost     0.30   
# Iteration 9999: Cost     0.30   
# ```

# <a name="2.7"></a>
# ### 2.7 Plotting the decision boundary
# 
# We will now use the final parameters from gradient descent to plot the linear fit. 
# We will use a helper function in the `utils.py` file to create this plot.

plot_decision_boundary(w, b, X_train, y_train)
# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

