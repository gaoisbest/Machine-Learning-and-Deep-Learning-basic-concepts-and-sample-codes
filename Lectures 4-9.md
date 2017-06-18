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
- Gradient check
```
# Numerically compute the gradient of loss function along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
# see Tensorflow gradient checking: https://www.tensorflow.org/versions/r0.11/api_docs/python/test/gradient_checking

def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
  """
  sample a few random elements and only return numerical
  in this dimensions.
  """

  for i in xrange(num_checks):
    ix = tuple([randrange(m) for m in x.shape]) # specfic point of W, corresponding to specfic loss of loss function f

    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evaluate f(x + h)
    x[ix] = oldval - h # increment by h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # reset

    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)
	
loss, grad = svm_loss_naive(W, X_dev, y_dev, 1e2)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 1e2)[0]
grad_numerical = grad_check_sparse(f, W, grad)
```
# Part solution of Assignment 1
## LinearClassifier.py
```
def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    
    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      random_indices = np.random.choice(num_train, batch_size)
      X_batch = X[random_indices,:]
      y_batch = y[random_indices]
      
	  # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform updates
	  self.W -= learning_rate*grad	  
	return loss_history

def predict(self, X):
    y_pred = np.argmax(np.dot(X, self.W), axis=1)
    return y_pred
```

  

