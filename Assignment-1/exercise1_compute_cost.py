# ### Exercise 1
# 
# Complete the `compute_cost` below to:
# 
# * Iterate over the training examples, and for each example, compute:
#     * The prediction of the model for that example 
#     $$
#     f_{wb}(x^{(i)}) =  wx^{(i)} + b 
#     $$
#    
#     * The cost for that example  $$cost^{(i)} =  (f_{wb} - y^{(i)})^2$$
#     
# 
# * Return the total cost over all examples
# $$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} cost^{(i)}$$
#   * Here, $m$ is the number of training examples and $\sum$ is the summation operator
# 
# If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

# In[23]:


# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    cost=0
    
    ### START CODE HERE ###
    for i in range(m):
        cost= cost+ (w*x[i] + b - y[i])**2
    total_cost=cost*(1/(2*m))
    ### END CODE HERE ### 

    return total_cost
compute_cost(x_train, y_train, 1, 2)


# <details>
#   <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
#     
#     
#    * You can represent a summation operator eg: $h = \sum\limits_{i = 0}^{m-1} 2i$ in code as follows:
#     
#     ```python 
#     h = 0
#     for i in range(m):
#         h = h + 2*i
#     ```
#   
#    * In this case, you can iterate over all the examples in `x` using a for loop and add the `cost` from each iteration to a variable (`cost_sum`) initialized outside the loop.
# 
#    * Then, you can return the `total_cost` as `cost_sum` divided by `2m`.
#    * If you are new to Python, please check that your code is properly indented with consistent spaces or tabs. Otherwise, it might produce a different output or raise an `IndentationError: unexpected indent` error. You can refer to [this topic](https://community.deeplearning.ai/t/indentation-in-python-indentationerror-unexpected-indent/159398) in our community for details.
# 
#     <details>
#           <summary><font size="2" color="darkblue"><b> Click for more hints</b></font></summary>
#         
#     * Here's how you can structure the overall implementation for this function
#     
#     ```python 
#     def compute_cost(x, y, w, b):
#         # number of training examples
#         m = x.shape[0] 
#     
#         # You need to return this variable correctly
#         total_cost = 0
#     
#         ### START CODE HERE ###  
#         # Variable to keep track of sum of cost from each example
#         cost_sum = 0
#     
#         # Loop over training examples
#         for i in range(m):
#             # Your code here to get the prediction f_wb for the ith example
#             f_wb = 
#             # Your code here to get the cost associated with the ith example
#             cost = 
#         
#             # Add to sum of cost for each example
#             cost_sum = cost_sum + cost 
# 
#         # Get the total cost as the sum divided by (2*m)
#         total_cost = (1 / (2 * m)) * cost_sum
#         ### END CODE HERE ### 
# 
#         return total_cost
#     ```
#     
#     * If you're still stuck, you can check the hints presented below to figure out how to calculate `f_wb` and `cost`.
#     
#     <details>
#           <summary><font size="2" color="darkblue"><b>Hint to calculate f_wb</b></font></summary>
#            &emsp; &emsp; For scalars $a$, $b$ and $c$ (<code>x[i]</code>, <code>w</code> and <code>b</code> are all scalars), you can calculate the equation $h = ab + c$ in code as <code>h = a * b + c</code>
#           <details>
#               <summary><font size="2" color="blue"><b>&emsp; &emsp; More hints to calculate f</b></font></summary>
#                &emsp; &emsp; You can compute f_wb as <code>f_wb = w * x[i] + b </code>
#            </details>
#     </details>
# 
#      <details>
#           <summary><font size="2" color="darkblue"><b>Hint to calculate cost</b></font></summary>
#           &emsp; &emsp; You can calculate the square of a variable z as z**2
#           <details>
#               <summary><font size="2" color="blue"><b>&emsp; &emsp; More hints to calculate cost</b></font></summary>
#               &emsp; &emsp; You can compute cost as <code>cost = (f_wb - y[i]) ** 2</code>
#           </details>
#     </details>
#         
#     </details>
# 
# </details>
# 
#     
# 

# You can check if your implementation was correct by running the following test code:

# In[24]:


# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Public tests
from public_tests import *
compute_cost_test(compute_cost)


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Cost at initial w:<b> 75.203 </td> 
#   </tr>
# </table>

# <a name="6"></a>
# ## 6 - Gradient descent 
# 
# In this section, you will implement the gradient for parameters $w, b$ for linear regression. 

# As described in the lecture videos, the gradient descent algorithm is:
# 
# $$\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \phantom {0000} b := b -  \alpha \frac{\partial J(w,b)}{\partial b} \newline       \; & \phantom {0000} w := w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{1}  \; & 
# \newline & \rbrace\end{align*}$$
# 
# where, parameters $w, b$ are both updated simultaniously and where  
# $$
# \frac{\partial J(w,b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{2}
# $$
# $$
# \frac{\partial J(w,b)}{\partial w}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) -y^{(i)})x^{(i)} \tag{3}
# $$
# * m is the number of training examples in the dataset
# 
#     
# *  $f_{w,b}(x^{(i)})$ is the model's prediction, while $y^{(i)}$, is the target value
# 
# 
# You will implement a function called `compute_gradient` which calculates $\frac{\partial J(w)}{\partial w}$, $\frac{\partial J(w)}{\partial b}$ 
