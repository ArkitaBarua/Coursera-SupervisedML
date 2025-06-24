# ### Exercise 5
# 
# Please complete the `compute_cost_reg` function below to calculate the following term for each element in $w$ 
# $$\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$
# 
# The starter code then adds this to the cost without regularization (which you computed above in `compute_cost`) to calculate the cost with regulatization.
# 
# If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

# In[109]:


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


# <details>
#   <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
#     
#     
# * Here's how you can structure the overall implementation for this function
#     ```python 
#        def compute_cost_reg(X, y, w, b, lambda_ = 1):
#    
#            m, n = X.shape
#     
#             # Calls the compute_cost function that you implemented above
#             cost_without_reg = compute_cost(X, y, w, b) 
#     
#             # You need to calculate this value
#             reg_cost = 0.
#     
#             ### START CODE HERE ###
#             for j in range(n):
#                 reg_cost_j = # Your code here to calculate the cost from w[j]
#                 reg_cost = reg_cost + reg_cost_j
#             reg_cost = (lambda_/(2 * m)) * reg_cost
#             ### END CODE HERE ### 
#     
#             # Add the regularization cost to get the total cost
#             total_cost = cost_without_reg + reg_cost
# 
#         return total_cost
#     ```
#   
#     If you're still stuck, you can check the hints presented below to figure out how to calculate `reg_cost_j` 
#     
#     <details>
#           <summary><font size="2" color="darkblue"><b>Hint to calculate reg_cost_j</b></font></summary>
#            &emsp; &emsp; You can use calculate reg_cost_j as <code>reg_cost_j = w[j]**2 </code> 
#     </details>
#         
#     </details>
# 
# </details>
# 
#     

# Run the cell below to check your implementation of the `compute_cost_reg` function.

# In[110]:


X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)

# UNIT TEST    
compute_cost_reg_test(compute_cost_reg)


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Regularized cost : <b></td>
#     <td> 0.6618252552483948 </td> 
#   </tr>
# </table>

# <a name="3.5"></a>
# ### 3.5 Gradient for regularized logistic regression
# 
# In this section, you will implement the gradient for regularized logistic regression.
# 
# 
# The gradient of the regularized cost function has two components. The first, $\frac{\partial J(\mathbf{w},b)}{\partial b}$ is a scalar, the other is a vector with the same shape as the parameters $\mathbf{w}$, where the $j^\mathrm{th}$ element is defined as follows:
# 
# $$\frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m}  \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})  $$
# 
# $$\frac{\partial J(\mathbf{w},b)}{\partial w_j} = \left( \frac{1}{m}  \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)} \right) + \frac{\lambda}{m} w_j  \quad\, \mbox{for $j=0...(n-1)$}$$
# 
# Compare this to the gradient of the cost function without regularization (which you implemented above), which is of the form 
# $$
# \frac{\partial J(\mathbf{w},b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)}) \tag{2}
# $$
# $$
# \frac{\partial J(\mathbf{w},b)}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)})x_{j}^{(i)} \tag{3}
# $$
# 
# 
# As you can see,$\frac{\partial J(\mathbf{w},b)}{\partial b}$ is the same, the difference is the following term in $\frac{\partial J(\mathbf{w},b)}{\partial w}$, which is $$\frac{\lambda}{m} w_j  \quad\, \mbox{for $j=0...(n-1)$}$$ 
# 
# 
# 
# 