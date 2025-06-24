# ### Exercise 2
# 
# Please complete the `compute_cost` function using the equations below.
# 
# Recall that for logistic regression, the cost function is of the form 
# 
# $$ J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \tag{1}$$
# 
# where
# * m is the number of training examples in the dataset
# 
# 
# * $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is - 
# 
#     $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$$
#     
#     
# *  $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$, which is the actual label
# 
# *  $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x^{(i)}} + b)$ where function $g$ is the sigmoid function.
#     * It might be helpful to first calculate an intermediate variable $z_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x^{(i)}} + b = w_0x^{(i)}_0 + ... + w_{n-1}x^{(i)}_{n-1} + b$ where $n$ is the number of features, before calculating $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(z_{\mathbf{w},b}(\mathbf{x}^{(i)}))$
# 
# Note:
# * As you are doing this, remember that the variables `X_train` and `y_train` are not scalar values but matrices of shape ($m, n$) and ($𝑚$,1) respectively, where  $𝑛$ is the number of features and $𝑚$ is the number of training examples.
# * You can use the sigmoid function that you implemented above for this part.
# 
# If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

# In[91]:


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
        f=sigmoid(z)
        total_cost=total_cost-y[𝑖]*math.log(f)-(1-y[𝑖])*math.log(1-f)
    
        
            
        
        
        
        
    
    ### END CODE HERE ### 

    return (1/m)*total_cost


# <details>
# <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
#     
# * You can represent a summation operator eg: $h = \sum\limits_{i = 0}^{m-1} 2i$ in code as follows:
# 
# ```python
#     h = 0
#     for i in range(m):
#         h = h + 2*i
# ```
# <br>
# 
# * In this case, you can iterate over all the examples in `X` using a for loop and add the `loss` from each iteration to a variable (`loss_sum`) initialized outside the loop.
# 
# * Then, you can return the `total_cost` as `loss_sum` divided by `m`.
# 
# * If you are new to Python, please check that your code is properly indented with consistent spaces or tabs. Otherwise, it might produce a different output or raise an `IndentationError: unexpected indent` error. You can refer to [this topic](https://community.deeplearning.ai/t/indentation-in-python-indentationerror-unexpected-indent/159398) in our community for details.
#      
# <details>
# <summary><font size="2" color="darkblue"><b> Click for more hints</b></font></summary>
#         
# * Here's how you can structure the overall implementation for this function
#         
# ```python
# def compute_cost(X, y, w, b, *argv):
#     m, n = X.shape
# 
#     ### START CODE HERE ###
#     loss_sum = 0 
#     
#     # Loop over each training example
#     for i in range(m): 
#         
#         # First calculate z_wb = w[0]*X[i][0]+...+w[n-1]*X[i][n-1]+b
#         z_wb = 0 
#         # Loop over each feature
#         for j in range(n): 
#             # Add the corresponding term to z_wb
#             z_wb_ij = # Your code here to calculate w[j] * X[i][j]
#             z_wb += z_wb_ij # equivalent to z_wb = z_wb + z_wb_ij
#         # Add the bias term to z_wb
#         z_wb += b # equivalent to z_wb = z_wb + b
#         
#         f_wb = # Your code here to calculate prediction f_wb for a training example
#         loss =  # Your code here to calculate loss for a training example
#         
#         loss_sum += loss # equivalent to loss_sum = loss_sum + loss
#         
#     total_cost = (1 / m) * loss_sum  
#     ### END CODE HERE ### 
#     
#     return total_cost
# ```
# <br>
# 
# If you're still stuck, you can check the hints presented below to figure out how to calculate `z_wb_ij`, `f_wb` and `cost`.
# 
# <details>
# <summary><font size="2" color="darkblue"><b>Hint to calculate z_wb_ij</b></font></summary>
#            &emsp; &emsp; <code>z_wb_ij = w[j]*X[i][j] </code>
# </details>
#         
# <details>
#           <summary><font size="2" color="darkblue"><b>Hint to calculate f_wb</b></font></summary>
#            &emsp; &emsp; $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(z_{\mathbf{w},b}(\mathbf{x}^{(i)}))$ where $g$ is the sigmoid function. You can simply call the `sigmoid` function implemented above.
#           <details>
#               <summary><font size="2" color="blue"><b>&emsp; &emsp; More hints to calculate f</b></font></summary>
#                &emsp; &emsp; You can compute f_wb as <code>f_wb = sigmoid(z_wb) </code>
#            </details>
# </details>
# 
# <details>
#           <summary><font size="2" color="darkblue"><b>Hint to calculate loss</b></font></summary>
#           &emsp; &emsp; You can use the <a href="https://numpy.org/doc/stable/reference/generated/numpy.log.html">np.log</a> function to calculate the log
#           <details>
#               <summary><font size="2" color="blue"><b>&emsp; &emsp; More hints to calculate loss</b></font></summary>
#               &emsp; &emsp; You can compute loss as <code>loss =  -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)</code>
# </details>
# </details>
#         
# </details>
# 
# </details>

# Run the cells below to check your implementation of the `compute_cost` function with two different initializations of the parameters $w$ and $b$

# In[92]:


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

# In[93]:


# Compute and display cost with non-zero w and b
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at test w and b (non-zeros): {:.3f}'.format(cost))


# UNIT TESTS
compute_cost_test(compute_cost)


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Cost at test w and b (non-zeros):<b></td>
#     <td> 0.218 </td> 
#   </tr>
# </table>
