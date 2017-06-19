# Basic concepts
- Neural network organized as a DAG. **Fully-connected layer** is the layer which neurons between two adjacent layers are fully pairwise connected, but neurons within a single layer share no connections. In practice, we should use as big of a neural network as your computational budget allows, and use other regularization techniques to control overfitting.

- **Universal approximator**: a single hidden layer neural network with the activation function can approximate any continuous function.

- A **single neuron** can be used to implement a **binary classifier** (e.g. logistic regression or binary SVM classifiers). From this view, logistic regression or SVMs are simply a special case of single-layer neural networks. 

- **Score function**, **loss function (data loss + regularization loss)** and **optimization**. Compute the gradient of a loss function with respect to its weights.

- Pipeline: *forward* computes the loss (i.e., repeated matrix multiplications interwoven with a bias offset and activation function), *backwards* computes the gradient, and perform *weights updating*.

- **Backpropagation**: recursive application of **chain rule**, that is, local gradient * the above gradient. 

- Patterns in backward flow
  - **Add gate**: local gradient is 1, distributes the above gradient equally to all of its inputs.
  - **Max gate**: routes the above gradient to exactly the max one of its inputs (local gradient is 1).
  - **Multiply gate**: local gradients are switched input values, times the above gradient. If one of the inputs is very small(`W`) and the other is very big(`X`), then the multiply gate will assign a relatively huge gradient to the small input (`W`) and a tiny gradient to the large input (`X`). During gradient descent, the gradient on the weights will be very large, then it should be come with lower learning rates. Therefore, **data preprocessing** is very necessary!
  - **Gradients add up at forks**: use `+=` instead of `=` to accumulate the gradient.
 
- Jacobian matrix and (Derivative of vector and matrix)[http://cs231n.stanford.edu/vecDerivs.pdf]

- Optimization
  - **Derivative** indicates the rate of change of a function with respect to that variable surrounding an infinitesimally small region `h` near a particular point. When `h` is very small, the function is well-approximated by a straight line, and the derivative is its slope.
  - **Gradient** is the vector of partial derivatives for each dimension.
  - **Gradient check**: always use the **analytic gradient**, check the results using **numerical gradient**.
  - Mini-batch gradient descent: common mini-batch sizes are 32, 64, 128 or 256 (many vectorized operation implementations work faster when their inputs are sized in powers of 2), usually based on memory constraints (if any).
  - **Learning rate** is crucial parameter.
 
- Activation function (i.e., non-linearity)
  - Sigmoid
    - Disadvantages: [vanishing gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) (saturate regime kill the gradients); non-zero centered outputs ([zig-zagging](https://zhuanlan.zhihu.com/p/25110450) gradient descent); exponential operations is expensive
  - Tanh
    - Disadvantages: vanishing gradient (saturate regime kill the gradients); non-zero centered outputs (zig-zagging gradient descent); exponential operations is expensive
  - [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
    - Advantages: fast convergence rate; cheaper operation; 
    - Disadvantages: [**Dying ReLU**](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks): the neuron state that neuron become inactive for all inputs and **a closed ReLU cannot update its input parameters**. In this state, no gradients flow backward through the neuron. This problem typically arises when the learning rate is set too high. It may be mitigated by using Leaky ReLUs instead.
    - A smooth approximation to ReLU is the **softplus** function `f(x)=ln(1 + e^x)`. The derivative of softplus is the **logistic function**.
  - Noisy ReLU
    - `f(x)=max(0,x+Y)`
  - Leaky ReLU
    - `f(x)=max(0.01x, x)`
  - Parameteric ReLU
    - `f(x)=max(ax, x)`
  - Maxout
  - Exponential LU
    - ELU try to make the mean activations closer to zero which speeds up learning. `f(x) = [x if x>=0 else a(e^x - 1)]`, where `a >= 0` is a hyper-parameter.


- Data preprocessing
  - PCA. 
     - If `Σ` is the covariance matrix of data `X`, PCA amounts to performing an eigendecomposition `Σ=UΛU^T`, where `U` is an orthogonal rotation matrix (`U^T = U^(-1)`) composed of eigenvectors of `Σ`, and `Λ` is a diagonal matrix with eigenvalues on the diagonal. Matrix `U^T` gives a rotation needed to de-correlate the data (i.e. maps the original features to principal components).
     - Depends on the ‘percentage of variance retained’ for setting top k component. Let `λ1,λ2,…,λn` be the eigenvalues of covariance matrix `Σ` (sorted in decreasing order), so that `λj` is the eigenvalue corresponding to the eigenvector `uj`. Then if we retain 2 principal components, the percentage of variance retained is given by: `(λ1 + λ2)/ (λ1 + λ2 + … + λn)`
  - Whitening. The goal of whitening is (i) the features are less correlated with each other, and (ii) the features all have the same variance. After the rotation `U^(T)(x−μ)` each component will have variance given by a corresponding eigenvalue. To make variances equal to 1, divide by the square root of Λ. Finally, the whitening transformation is `Λ^(−1/2)U^(T)(x−μ)`. This data now has covariance equal to the identity matrix I.Reference: http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/



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

- Sigmoid and its derivative
```
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 500)
sigmoid = lambda x: 1.0/(1.0 + np.exp(-x))
sigmoid_der = lambda x: sigmoid(x) * (1 - sigmoid(x))
plt.plot(x, sigmoid(x))
plt.plot(x, sigmoid_der(x))
```

- Tanh 
```
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 500)
tanh = lambda x: (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))
plt.plot(x, tanh(x))
```

- Comparison of variants of ReLU and softplus
```
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-2, 2, 500)
relu = np.maximum(0, x)
leaky_relu = np.maximum(0.01*x, x) # a=0.01
softplus = np.log(1 + np.exp(x))
elu = 0.02*(np.exp(x) - 1) # a = 0.02

plt.plot(x, relu)
plt.plot(x, leaky_relu)
plt.plot(x, softplus)
plt.plot(x, elu)
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

  

