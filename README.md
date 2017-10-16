# Basic concepts and sample codes about ML and DL

Since famous [CS231n](http://cs231n.stanford.edu/) and [Andrew Ng's new DL course](https://www.coursera.org/specializations/deep-learning) are both introduce basic concepts about ML and DL, so I combine them together. 

Sample projects list:
- [Vectorized logistic regression](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Logistic%20regression/Logistic_regression_vectorized.py)

![](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Logistic%20regression/Logistic%20regression.png)

- [Vectorized neural network](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Neural%20network/Neural_network_vectorized.py)

![](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Neural%20network/Neural%20network.png)


# Classical questions
## 1. Why is logistic regression a generalized linear model ? 
Suppose that the logistic regression, `p = sigmoid(z)`:
- The input `z`, i.e., `z = WX + b`, is a linear function of x.
- Log-odds, i.e., `log (p/(1-p)) = WX`, is a linear function of parameters `W`.
- **Decision boundary**, i.e., `WX = 0`, is linear. This does not mean LR can only handle linear-separable data. Actually, we can convert low-dimensional data to high-dimensional data with the help of **feature mapping**. And the data are linear-separable in the high-dimensional space.
- Both logistic regression and softmax regression can be modeled by exponential family distribution.

Definition: **linear** means linear in parameters `W` but not in `X`.

References:  
https://www.quora.com/Why-is-logistic-regression-considered-a-linear-model  
https://stats.stackexchange.com/questions/93569/why-is-logistic-regression-a-linear-classifier  
https://stats.stackexchange.com/questions/88603/why-is-logistic-regression-a-linear-model  
http://cs229.stanford.edu/notes/cs229-notes1.pdf  

Logistic regression application in Meituan:  
https://tech.meituan.com/intro_to_logistic_regression.html

## 2. L1 and L2 regularization
- Why are the shape of the L1 norm and L2 norm diamond and circle like respectively? 
See reference [1].
- Why does L1 lead to sparse weights and L2 lead to small distributed weights?
L1 regularization helps performing **feature selection**.
- L1 or L2, which one is perfered in differenct scenario?
Both are used for solve over-fitting.
L1 norm is not differentiable at zero point, see picture from [3]:
Someone said the L2 regularization always the best choice.
- L2 regularization Implementation
  - forward propagation computes cost
	```
	# suppose that there are sigler hidden layer neural network
	# W1 and W2 are parameters for input X and hidden neurons
	L2_regularization_cost = 1.0 / m * lambd / 2.0 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	cost = cross_entropy_cost + L2_regularization_cost
	```
  - back propagation comptutes gradients
	```
	dZ2 = A2 - Y # the cross_entropy loss
    dW2 = 1./m * np.dot(dZ2, A1.T) + W2 * lambd / m
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0)) # relu activation
    dW1 = 1./m * np.dot(dZ1, X.T) + W1 * lambd / m
	db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
	```
- Lasso and ridge regression
- Elastic net 
- L0 norm: the number of non-zero elements. L1 norm is the most convex approximation to L0.

Since we add regularization term, the cost cannot be zero at anytime.

references:  
[1] https://www.quora.com/Why-does-an-L1-norm-unit-ball-have-diamond-shaped-geometry#!n=12
[2]https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when  
[3] https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models
[4] https://feature.engineering/regularization/  


Classical problem solutions (to be done):
- [Imbalanced training data]()
- [Missing data]()
- [Feature selection]()
- [Evaluation matrics]()
- ...

