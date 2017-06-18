# Basic concepts
- **Score function**, **loss function (data loss + regularization loss)** and **optimization**. Compute the gradient of a loss function with respect to its weights.

- Pipeline: forward computes the loss, backwards computes the gradient, and perform weights updating.

- Optimization
  - **Derivative** indicates the rate of change of a function with respect to that variable surrounding an infinitesimally small region `h` near a particular point. When `h` is very small, the function is well-approximated by a straight line, and the derivative is its slope.
  - **Gradient** is the vector of partial derivatives for each dimension.
  - **Gradient check**: always use the **analytic gradient**, check the results using **numerical gradient**.
  - Mini-batch gradient descent: common mini-batch sizes are 32, 64, 128 or 256 (many vectorized operation implementations work faster when their inputs are sized in powers of 2), usually based on memory constraints (if any).
  - **Learning rate** is crucial parameter.
  
- **Backpropagation**: recursive application of **chain rule**, that is, local gradient * the above gradient. 

- Patterns in backward flow
  - **Add gate**: local gradient is 1, distributes the above gradient equally to all of its inputs.
  - **Max gate**: routes the above gradient to exactly the max one of its inputs (local gradient is 1).
  - **Multiply gate**: local gradients are switched input values, times the above gradient. If one of the inputs is very small(`W`) and the other is very big(`X`), then the multiply gate will assign a relatively huge gradient to the small input (`W`) and a tiny gradient to the large input (`X`). During gradient descent, the gradient on the weights will be very large, then it should be come with lower learning rates. Therefore, **data preprocessing** is very necessary!
  - **Gradients add up at forks**: use `+=` instead of `=` to accumulate the gradient.
 
- Jacobian matrix

- Data preprocessing
k components, 
To decide how to set k, we will usually look at the ‘percentage of variance retained’ for different values of k. More generally, let λ1,λ2,…,λn be the eigenvalues of Σ (sorted in decreasing order), so that λj is the eigenvalue corresponding to the eigenvector uj. Then if we retain 2 principal components, the percentage of variance retained is given by: (λ1 + λ2)/ (λ1 + λ2 + … + λn)

The goal of whitening is (i) the features are less correlated with each other, and (ii) the features all have the same variance.

Reference:
http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/


If Σ is the covariance matrix of data X, PCA amounts to performing an eigendecomposition Σ=UΛU^T, where U is an orthogonal rotation matrix (U^T = U^(-1)) composed of eigenvectors of Σ, and Λ is a diagonal matrix with eigenvalues on the diagonal. Matrix U^T gives a rotation needed to de-correlate the data (i.e. maps the original features to principal components). The corvariance matrix of Matrix U^T is **diagonal** matrix.

Third, after the rotation each component will have variance given by a corresponding eigenvalue. So to make variances equal to 1, you need to divide by the square root of Λ.

All together, the whitening transformation is Λ^(−1/2)U^(T)(x−μ). This data now has covariance equal to the identity matrix I.
-

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

- 11 lines of nerual network

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

  

