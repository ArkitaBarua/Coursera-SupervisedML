# ### Exercise 2
# 
# Please complete the `compute_gradient` function to:
# 
# * Iterate over the training examples, and for each example, compute:
#     * The prediction of the model for that example 
#     * The gradient for the parameters $w, b$ from that example 
#     * Return the total gradient update from all the examples

# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    ### START CODE HERE ###
    for i in range(m):
        dj_dw+=((w*x[i]+b-y[i])*x[i])/m
        dj_db+=((w*x[i]+b-y[i]))/m
    ### END CODE HERE ### 
        
    return dj_dw, dj_db


# Run the cells below to check your implementation of the `compute_gradient` function with two different initializations of the parameters $w$,$b$.
# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

#compute_gradient_test(compute_gradient)

# Now let's run the gradient descent algorithm implemented above on our dataset.
# 
# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Gradient at initial , b (zeros)<b></td>
#     <td> -65.32884975 -5.83913505154639</td> 
#   </tr>
# </table>

# In[28]:


# Compute and display cost and gradient with non-zero w
test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Gradient at test w<b></td>
#     <td> -47.41610118 -4.007175051546391</td> 
#   </tr>
# </table>

# <a name="2.6"></a>
# ### 2.6 Learning parameters using batch gradient descent 
# 
# You will now find the optimal parameters of a linear regression model by using batch gradient descent. Recall batch refers to running all the examples in one iteration.
# - You don't need to implement anything for this part. Simply run the cells below. 
# 
# - A good way to verify that gradient descent is working correctly is to look
# at the value of $J(w,b)$ and check that it is decreasing with each step. 
# 
# - Assuming you have implemented the gradient and computed the cost correctly and you have an appropriate value for the learning rate alpha, $J(w,b)$ should never increase and should converge to a steady value by the end of the algorithm.

# In[29]:


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing


# Now let's run the gradient descent algorithm above to learn the parameters for our dataset.
# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b> w, b found by gradient descent<b></td>
#     <td> 1.16636235 -3.63029143940436</td> 
#   </tr>
# </table>

# We will now use the final parameters from gradient descent to plot the linear fit.  
# To calculate the predictions on the entire dataset, we can loop through all the training examples and calculate the prediction for each example. This is shown in the code block below.


m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b


# We will now plot the predicted values to see the linear fit.
# Plot the linear fit
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')


# Your final values of $w,b$ can also be used to make predictions on profits. Let's predict what the profit would be in areas of 35,000 and 70,000 people. 
# 
# - The model takes in population of a city in 10,000s as input. 
# 
# - Therefore, 35,000 people can be translated into an input to the model as `np.array([3.5])`
# 
# - Similarly, 70,000 people can be translated into an input to the model as `np.array([7.])`
# 


predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b> For population = 35,000, we predict a profit of<b></td>
#     <td> $4519.77 </td> 
#   </tr>
#   
#   <tr>
#     <td> <b> For population = 70,000, we predict a profit of<b></td>
#     <td> $45342.45 </td> 
#   </tr>
# </table>
