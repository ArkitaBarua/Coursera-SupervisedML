# ### Exercise 1
# 
# Complete the `compute_cost` below to:
# 
# * Iterate over the training examples, and for each example, compute:
#     * The prediction of the model for that example 
#     * The cost for that example 
#     * Return the total cost over all examples
# 


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


# You can check if your implementation was correct by running the following test code:
# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Public tests
#from public_tests import *
#compute_cost_test(compute_cost)


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Cost at initial w:<b> 75.203 </td> 
#   </tr>
# </table>
