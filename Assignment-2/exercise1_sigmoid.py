# ### Exercise 1
# Please complete  the `sigmoid` function 
# 
# Note that 
# - `z` is not always a single number, but can also be an array of numbers. 
# - If the input is an array of numbers, we'd like to apply the sigmoid function to each value in the input array.

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

# - For large positive values of x, the sigmoid should be close to 1, while for large negative values, the sigmoid should be close to 0. 
# - Evaluating `sigmoid(0)` should give you exactly 0.5. 

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
