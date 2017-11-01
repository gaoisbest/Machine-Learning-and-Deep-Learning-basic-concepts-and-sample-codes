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
#### Why are the shape of the L1 norm and L2 norm diamond like and circle like respectively? 
See reference [1].

#### Why does L1 lead to sparse weights and L2 lead to small distributed weights?  
L0 norm is the number of non-zero elements, which has sparse property.  
**L1 norm is the best convex approximation to L0 norm**. We can view L1 norm as a compromise between the L0 and L2 norms, inheriting the sparsity-inducing property from the former and convexity from the latter [2].  
The L1 norm despite being convex, is **not everywhere differentiable** (unlike the L2 norm) [2], see picture from [3].  
![](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Andrew_Ng_images/Class_2_week_1/L1_derivative.png)  
Traditional gradient descent cannot be used to optimize L1 regularized models, therefore more advanced techniques like **proximal gradient** or primal-dual methods like ADMM are required [2].  
L1 regularization helps performing **feature selection**.  
Since **squaring a number punishes large values more than it punishes small values** [2], L2 regularization leads to small distributed weights.

#### L1 or L2, which one is perfered in differenct scenario?
Both are used for solve over-fitting.  
In Bayesian view:  
L1 regularization is equivalent to MAP estimation using **Laplacian prior**.  
L2 regularization is equivalent to MAP estimation using **Gaussian prior**.  
Always try L2 regularization first, since it will give you the best result [2].

#### L2 regularization Implementation
  - forward propagation computes cost
	```
	# suppose that there are sigler hidden layer neural network
	# W1 and W2 are parameters for input X and hidden neurons
	# Since we add regularization term, the cost cannot be zero at anytime.
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
#### Lasso (with L1 regularization) and ridge regression (with L2 regularization).  
The following picture is from [2].  
![](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Andrew_Ng_images/Class_2_week_1/Lasso_and_ridge.png)  

#### Elastic net (combine L1 and L2)  
```
# from http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
1 / (2 * n_samples) * ||y - Xw||^2_2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
```

References:  
[1] https://www.quora.com/Why-does-an-L1-norm-unit-ball-have-diamond-shaped-geometry#!n=12  
[2] https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when  
[3] https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models  
[4] https://feature.engineering/regularization/  

## 3. Exploding/vanishing gradients  
http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf  
http://www.deeplearningbook.org/contents/rnn.html  

#### Vanishing gradients [1, 3]:
- Results: early layers are **converged slower** than later layers. 
- Reason: `Sigmoid` and `tanh` activations suffer from vanishing gradients. But ReLU activation does not have this problem.
s property of 'squashing' the input space into a small region. 
- Solutions:  
[1] ReLU or Leaky ReLU. ReLU can have **dying** states (caused by i.e., large negative bias), whose both outputs and gradients are zero. Leaky ReLU solves this problem.  
[2] Proper weights initialization can reduce the effect of vanishing gradients.  
[3] LSTM or GRU.
- It is a more serious problem than exploding gradients.

#### Exploding gradients [2, 3]:
- Results: gradients and cost become NaN
- Reason: large weights and derivative of activation multiplication during back propagation. It is particularly occured in RNNs.
- Solution: gradient clipping
```
if norm(gradients) > max_gradients_norm:
    gradients *= max_gradients_norm/norm(gradients)
```

#### Why LSTM resistant to exploding and vanishing gradients?
- If the forget gate is on and the input and output gates are off, it just passes the memory cell gradients through unmodified at each time step [4].

References:  
[1] https://ayearofai.com/rohan-4-the-vanishing-gradient-problem-ec68f76ffb9b  
[2] https://www.quora.com/Why-is-it-a-problem-to-have-exploding-gradients-in-a-neural-net-especially-in-an-RNN  
[3] http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/  
[4] http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf



## 4. Batch normalization

## 5. Evaluation metrics
### 5.1 Accuracy, precision, recall and F1
Consider a scenario that predicting the gender of the user (i.e., male or female). This is a classical **binary classification** prediction problem. Let predicted 1 (i.e., positive) indicates male and predicted 0 (i.e., negative) indicates female.  
**Confusion matrix**:  

|               | Predicted positive |   Predicted negative  |
|   :---:       | :---:             |     :---:                |
|Real positive |      TP             |         FN               |
|Real negative |       FP            |      TN   |  

**Accuracy** = (TP + TN) / (TP+FP+FN+TN)  
**Error** = (FP + FN) / (TP+FP+FN+TN) = 1 - accuracy  
**Precision** = TP / (TP + FP) What's the correct proportion of predicted positive ?  
**Recall** = TP / (TP + FN) What's the pick up proportion of all positive obsverations ?  
**F1** = 2 * Precision * Recall / (Precision + Recall)  

Which model is better?  
Model 1:  

|               | Predicted 1 |   Predicted 0  |
|   :---:       | :---:               |     :---:                |
|Real 1 |      4             |         2              |
|Real 0|       3            |      1   |  

Model 2:  

|               | Predicted 1 |   Predicted 0  |
|   :---:       | :---:               |     :---:                |
|Real 1 |      6            |         0               |
|Real 0 |       4            |      0   |  


Model 1: accuracy = 92%.  
Model 2: accuracy = 95%.  
Model 2 has higher accuracy than model 1, but model 2 is useless. This is called [accuracy paradox](https://en.wikipedia.org/wiki/Accuracy_paradox), which means the model with higher accuracy may not have better generalization power.   
In general, when **TP < FP**, the accuracy will always increase when we change the classifier to always output **'negative'**. Conversely, when **TN < FN**, the same will happen when we change the classifier to always output **'positive'** [1].  

### 5.2 ROC, AUC

References:  
[1] https://tryolabs.com/blog/2013/03/25/why-accuracy-alone-bad-measure-classification-tasks-and-what-we-can-do-about-it/  
[2] https://www.zhihu.com/question/30643044  

Classical problem solutions (to be done):
- [Imbalanced training data](https://svds.com/learning-imbalanced-classes/)
- [Missing data]()
- [Feature selection]()
- ...

