# ### Exercise 6
# 
# Please complete the `compute_gradient_reg` function below to modify the code below to calculate the following term
# 
# $$\frac{\lambda}{m} w_j  \quad\, \mbox{for $j=0...(n-1)$}$$
# 
# The starter code will add this term to the $\frac{\partial J(\mathbf{w},b)}{\partial w}$ returned from `compute_gradient` above to get the gradient for the regularized cost function.
# 
# 
# If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

# In[111]:


# UNQ_C6
def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    """
    Computes the gradient for logistic regression with regularization
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    m, n = X.shape
    
    dj_db, dj_dw = compute_gradient(X, y, w, b)

    ### START CODE HERE ###     
    for i in range(n):
        dj_dw[i]+=(lambda_/m)*w[i]
        
    ### END CODE HERE ###         
        
    return dj_db, dj_dw


# <details>
#   <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
#     
#     
# * Here's how you can structure the overall implementation for this function
#     ```python 
#     def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
#         m, n = X.shape
#     
#         dj_db, dj_dw = compute_gradient(X, y, w, b)
# 
#         ### START CODE HERE ###     
#         # Loop over the elements of w
#         for j in range(n): 
#             
#             dj_dw_j_reg = # Your code here to calculate the regularization term for dj_dw[j]
#             
#             # Add the regularization term  to the correspoding element of dj_dw
#             dj_dw[j] = dj_dw[j] + dj_dw_j_reg
#         
#         ### END CODE HERE ###         
#         
#         return dj_db, dj_dw
#     ```
#   
#     If you're still stuck, you can check the hints presented below to figure out how to calculate `dj_dw_j_reg` 
#     
#     <details>
#           <summary><font size="2" color="darkblue"><b>Hint to calculate dj_dw_j_reg</b></font></summary>
#            &emsp; &emsp; You can use calculate dj_dw_j_reg as <code>dj_dw_j_reg = (lambda_ / m) * w[j] </code> 
#     </details>
#         
#     </details>
# 
# </details>
# 
#     
# 

# Run the cell below to check your implementation of the `compute_gradient_reg` function.

# In[112]:


X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1) 
initial_w  = np.random.rand(X_mapped.shape[1]) - 0.5 
initial_b = 0.5
 
lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )

# UNIT TESTS    
compute_gradient_reg_test(compute_gradient_reg)


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>dj_db:</b>0.07138288792343</td> </tr>
#   <tr>
#       <td> <b> First few elements of regularized dj_dw:</b> </td> </tr>
#    <tr>
#    <td> [[-0.010386028450548], [0.011409852883280], [0.0536273463274], [0.003140278267313]] </td> 
#   </tr>
# </table>

# <a name="3.6"></a>
# ### 3.6 Learning parameters using gradient descent
# 
# Similar to the previous parts, you will use your gradient descent function implemented above to learn the optimal parameters $w$,$b$. 
# - If you have completed the cost and gradient for regularized logistic regression correctly, you should be able to step through the next cell to learn the parameters $w$. 
# - After training our parameters, we will use it to plot the decision boundary. 
# 
# **Note**
# 
# The code block below takes quite a while to run, especially with a non-vectorized version. You can reduce the `iterations` to test your implementation and iterate faster. If you have time later, run for 100,000 iterations to see better results.

# In[113]:


# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ (you can try varying this)
lambda_ = 0.01    

# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                    compute_cost_reg, compute_gradient_reg, 
                                    alpha, iterations, lambda_)


# <details>
# <summary>
#     <b>Expected Output: Cost < 0.5  (Click for details)</b>
# </summary>
# 
# ```
# # Using the following settings
# #np.random.seed(1)
# #initial_w = np.random.rand(X_mapped.shape[1])-0.5
# #initial_b = 1.
# #lambda_ = 0.01;                                          
# #iterations = 10000
# #alpha = 0.01
# Iteration    0: Cost     0.72   
# Iteration 1000: Cost     0.59   
# Iteration 2000: Cost     0.56   
# Iteration 3000: Cost     0.53   
# Iteration 4000: Cost     0.51   
# Iteration 5000: Cost     0.50   
# Iteration 6000: Cost     0.48   
# Iteration 7000: Cost     0.47   
# Iteration 8000: Cost     0.46   
# Iteration 9000: Cost     0.45   
# Iteration 9999: Cost     0.45       
#     
# ```

# <a name="3.7"></a>
# ### 3.7 Plotting the decision boundary
# To help you visualize the model learned by this classifier, we will use our `plot_decision_boundary` function which plots the (non-linear) decision boundary that separates the positive and negative examples. 
# 
# - In the function, we plotted the non-linear decision boundary by computing the classifier’s predictions on an evenly spaced grid and then drew a contour plot of where the predictions change from y = 0 to y = 1.
# 
# - After learning the parameters $w$,$b$, the next step is to plot a decision boundary similar to Figure 4.
# 
# <img src="images/figure 4.png"  width="450" height="450">

# In[114]:


plot_decision_boundary(w, b, X_mapped, y_train)
# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()


# <a name="3.8"></a>
# ### 3.8 Evaluating regularized logistic regression model
# 
# You will use the `predict` function that you implemented above to calculate the accuracy of the regularized logistic regression model on the training set

# In[115]:


#Compute accuracy on the training set
p = predict(X_mapped, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Train Accuracy:</b>~ 80%</td> </tr>
# </table>

# **Congratulations on completing the final lab of this course! We hope to see you in Course 2 where you will use more advanced learning algorithms such as neural networks and decision trees. Keep learning!**

# <details>
#   <summary><font size="2" color="darkgreen"><b>Please click here if you want to experiment with any of the non-graded code.</b></font></summary>
#     <p><i><b>Important Note: Please only do this when you've already passed the assignment to avoid problems with the autograder.</b></i>
#     <ol>
#         <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “Edit Metadata”</li>
#         <li> Hit the “Edit Metadata” button next to the code cell which you want to lock/unlock</li>
#         <li> Set the attribute value for “editable” to:
#             <ul>
#                 <li> “true” if you want to unlock it </li>
#                 <li> “false” if you want to lock it </li>
#             </ul>
#         </li>
#         <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “None” </li>
#     </ol>
#     <p> Here's a short demo of how to do the steps above: 
#         <br>
#         <img src="https://lh3.google.com/u/0/d/14Xy_Mb17CZVgzVAgq7NCjMVBvSae3xO1" align="center" alt="unlock_cells.gif">
# </details>
