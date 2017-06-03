# Basic concepts
- **Score function**, **loss function** and **optimization**. Compute the gradient of a loss function with respect to its weights.
- Forward computes the loss, backwards computes the gradient, and perform weights updating.
- Backpropagation: local gradient * the above gradient. 
- Patterns in backward flow
  - **Add gate**: local gradient is 1, distributes the above gradient equally to all of its inputs.
  - **Max gate**: routes the above gradient to exactly the max one of its inputs (local gradient is 1).
  - **Multiply gate**: local gradients are switched input values, times the above gradient. If one of the inputs is very small(`W`) and the other is very big(`X`), then the multiply gate will assign a relatively huge gradient to the small input (`W`) and a tiny gradient to the large input (`X`). During gradient descent, the gradient on the weights will be very large, then it should be come with lower learning rates. Therefore, **data preprocessing** is very necessary!
  - Gradients add up at forks: use `+=` instead of `=` to accumulate the gradient.
  
# Codes
# Part solution of Assignment 2

  

