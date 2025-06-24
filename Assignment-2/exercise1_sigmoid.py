# ### Exercise 1
# Please complete  the `sigmoid` function to calculate
# 
# $$g(z) = \frac{1}{1+e^{-z}}$$
# 
# Note that 
# - `z` is not always a single number, but can also be an array of numbers. 
# - If the input is an array of numbers, we'd like to apply the sigmoid function to each value in the input array.
# 
# If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

# In[88]:


# UNQ_C1
# GRADED FUNCTION: sigmoid
import numpy as np
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
          
    ### START CODE HERE ### 
    g=1/(1+np.exp(-z))
    
    ### END SOLUTION ###  
    
    return g


# <details>
#   <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
#        
#    * `numpy` has a function called [`np.exp()`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html), which offers a convinient way to calculate the exponential ( $e^{z}$) of all elements in the input array (`z`).
#  
# <details>
#           <summary><font size="2" color="darkblue"><b> Click for more hints</b></font></summary>
#         
#   - You can translate $e^{-z}$ into code as `np.exp(-z)` 
#     
#   - You can translate $1/e^{-z}$ into code as `1/np.exp(-z)` 
#     
#     If you're still stuck, you can check the hints presented below to figure out how to calculate `g` 
#     
#     <details>
#           <summary><font size="2" color="darkblue"><b>Hint to calculate g</b></font></summary>
#         <code>g = 1 / (1 + np.exp(-z))</code>
#     </details>
# 
# 
# </details>

# When you are finished, try testing a few values by calling `sigmoid(x)` in the cell below. 
# - For large positive values of x, the sigmoid should be close to 1, while for large negative values, the sigmoid should be close to 0. 
# - Evaluating `sigmoid(0)` should give you exactly 0.5. 
# 

# In[89]:


# Note: You can edit this value
value = 0

print (f"sigmoid({value}) = {sigmoid(value)}")


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>sigmoid(0)<b></td>
#     <td> 0.5 </td> 
#   </tr>
# </table>
#     
# - As mentioned before, your code should also work with vectors and matrices. For a matrix, your function should perform the sigmoid function on every element.

# In[90]:


print ("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))

# UNIT TESTS
from public_tests import *
sigmoid_test(sigmoid)


# **Expected Output**:
# <table>
#   <tr>
#     <td><b>sigmoid([-1, 0, 1, 2])<b></td> 
#     <td>[0.26894142        0.5           0.73105858        0.88079708]</td> 
#   </tr>    
#   
# </table>
