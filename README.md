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

### What's the difference between linear and non-linear regression ? 
- It is **wrong** that **linear regression model generates straight lines and nonlinear regression model curvature**. Actually, both linear and non-linear models can fit curves.  
- Linear means **linear in parameters `W` but not in `X`**. For example, both functions `f_1 = w_1x_1 + w_2x_2 + b` and `f_2 = w_1x_1 + w_2x_2 + w_3x_2^{2} + b` are linear functions. The most common operation is adding **polynomial terms** (i.e., quadratic term or cubic term) in linear model.
- Example of non-linear function `f_3 = w_1x^{w_1}`
- **`log` transform** can be used to convert nonlinear function `y=e^{b}x_1^{w_1}x_2^{w_2}` to linear function `lny = b + w_1lnx_1 + w_2lnx_2`

References:  
https://www.quora.com/Why-is-logistic-regression-considered-a-linear-model  
https://stats.stackexchange.com/questions/93569/why-is-logistic-regression-a-linear-classifier  
https://stats.stackexchange.com/questions/88603/why-is-logistic-regression-a-linear-model  
http://cs229.stanford.edu/notes/cs229-notes1.pdf  
http://statisticsbyjim.com/regression/difference-between-linear-nonlinear-regression-models/  
http://statisticsbyjim.com/regression/curve-fitting-linear-nonlinear-regression/  
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
#### Vanishing gradients [1, 3]:
- Results: early layers are **converged slower** than later layers. 
- Reason: `sigmoid` and `tanh` activations suffer from vanishing gradients. But `ReLU` activation does not have this problem.
- Solutions:
	- **Activation function**.`ReLU` or `Leaky ReLU`. `ReLU` can have **dying** states (caused by i.e., large learning rate or large negative bias), whose both outputs and gradients are zero. `Leaky ReLU` solves this problem. Variants of `Leaky ReLU` is `randomized leaky ReLU (RReLU)`, `parametric leaky ReLU (PReLU)`. `exponential linear unit (ELU)`.  
	**ELU > Leaky ReLU > ReLU > tanh > sigmoid** [5].
	- **Weights initialization**. i.e., `Xavier` initialization (**Two goal**: the **outputs variance** of each layer is equal to the **inputs variance**; the **gradients variance** before and after flowing through a layer is equal) for `sigmoid` and `tanh`, `He` initialization for `ReLU` and `Leaky ReLU`. 
	- **Batch Normalization**. Address vanishing or exploding gradients problem during training [6]. For more detailed, see 4. Batch Normalization.
		- Zero-centering + normalizing + scaling + shifting
		- At test time, use the whole training set's mean and standard deviation.
- Implementation [5]
```
# xavier
tf.contrib.layers.fully_connected(inputs, num_outputs, weights_initializer=initializers.xavier_initializer())

# He
he_init = tf.contrib.layers.variance_scaling_initializer() # default is He initialization that only consider fan-in
tf.contrib.layers.fully_connected(inputs, num_outputs, weights_initializer=he_init)

# elu
tf.contrib.layers.fully_connected(inputs, num_outputs, activation_fn=tf.nn.elu)

# leaky relu
def leaky_relu(z, name=None):
    return tf.maximum(0.01*z, z, name=name)
tf.contrib.layers.fully_connected(inputs, num_outputs, activation_fn=leaky_relu)
```
![](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Neural%20network/Xavier_He_initialization.png)

#### Exploding gradients [2, 3]:
- Results: gradients and cost become `NaN`.
- Reason: large weights and derivative of activation multiplication during back propagation. It is particularly occured in RNNs.
- Solution: **gradients clipping**.
- Implementation:  
```
# pseudo code
if norm(gradients) > max_gradients_norm:
    gradients *= max_gradients_norm/norm(gradients)

# real code
variables = tf.trainable_variables()
gradients = tf.gradients(ys=cost, xs=variables)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.clip_norm)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
optimize = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, variables), global_step=self.global_step)
```

#### Why LSTM resistant to exploding and vanishing gradients?
- If the forget gate is on and the input and output gates are off, it just passes the memory cell gradients through unmodified at each time step [4].
- CEC (Constant Error Carrousel) mechanism with gate [7].

References:  
[1] https://ayearofai.com/rohan-4-the-vanishing-gradient-problem-ec68f76ffb9b  
[2] https://www.quora.com/Why-is-it-a-problem-to-have-exploding-gradients-in-a-neural-net-especially-in-an-RNN  
[3] http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/  
[4] http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf  
[5] Hands on machine learning with Scikit-Learn and TensorFlow p278, P281  
[6] https://www.zhihu.com/question/38102762  
[7] https://www.zhihu.com/question/34878706

## 4. Batch Normalization
- He initialization and ELU can reduce the vanishing gradients **at the begining of training**.
- Address the vanishing gradients during **training**.
- **Batch** means evaluating the mean and standard deviation of the inputs over current mini-batch.
- At test time, use the whole training set's mean and standard deviation.
- BN can reduce vanishing gradients problem, less sensitive to the weight initialization, reduce the need for other regularization techniques (such as dropout).

![](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Neural%20network/Batch_normalization_formula.png)  


## 5. Nerual Networks
### Back propagation
**Chain rule with memorization.**

### How to choose the number of hidden layers ? 
DNN could extract features layer by layer, and it has a **hierarchical architecture**. For many problems, **one or two hidden layers** will works fine. For complex problem, you can gradually increase the number of hidden layers, until overfitting occurs.

### How to set the number of neurons per hidden layer ? 
A common strategy is to size the number of neurons to form a funnel (i.e., **deeper layer has fewer neruons**). The analogy is many low-level features are coalesce into fewer high-level features.  
A simple approach is to pick a complex model with early stopping to prevent from overfitting.  

References:  
[1] Hands on machine learning with Scikit-Learn and TensorFlow p271
### How does batch size influence training speed and model accuracy ?
Batch gradient descent
- slow
- may converge to local minimum, and yield worse performance
- suffer from GPU memory limitation

Stochastic gradient descent
- fast
- not stable

Mini-batch gradient descent
- fast as SGD (matrix operation with GPU)
- escape from local minimum and more stable

### XOR solution
![](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Neural%20network/XOR.png)  

One hidden layer with two neurons. The activation in above image is step function.

## 6. Evaluation metrics
### 6.1 Accuracy, precision, recall and F1
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

### 6.2 ROC, AUC

References:  
[1] https://tryolabs.com/blog/2013/03/25/why-accuracy-alone-bad-measure-classification-tasks-and-what-we-can-do-about-it/  
[2] https://www.zhihu.com/question/30643044  

Classical problem solutions (to be done):
- [Imbalanced training data](https://svds.com/learning-imbalanced-classes/)
- [Missing data]()
- [Feature selection]()
- ...

## 7. Loss function
- Least mean square: regression
- Cross entropy (shortcut of KL divergence, relative entropy): logistic regression
- Exponential loss: Adaboost
- Hinge loss: SVM
