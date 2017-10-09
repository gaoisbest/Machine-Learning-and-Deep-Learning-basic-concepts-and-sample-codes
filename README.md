# Basic concepts and sample codes about ML and DL

Since famous [CS231n](http://cs231n.stanford.edu/) and [Andrew Ng's new DL course](https://www.coursera.org/specializations/deep-learning) are both introduce basic concepts about ML and DL, so I combine them together. 

Sample projects list:
- [Vectorized logistic regression](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Logistic%20regression/Logistic_regression_vectorized.py)

![](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Logistic%20regression/Logistic%20regression.png)

- [Vectorized neural network](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Neural%20network/Neural_network_vectorized.py)

![](https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Neural%20network/Neural%20network.png)


# Classical questions
## Why is logistic regression a generalized linear model ? 
Suppose that the logistic regression, `p = sigmoid(z)`:
- The input `z`, i.e., `z = WX + b`, is a linear function of x.
- Log-odds, i.e., `log (p/(1-p)) = WX`, is a linear function of parameters `W`.
- **Decision boundary**, i.e., `WX = 0`, is linear.
- Both logistic regression and softmax regression can be modeled by exponential family distribution.

Definition: **linear** means linear in parameters `W` but not in `X`.

references:  
https://www.quora.com/Why-is-logistic-regression-considered-a-linear-model  
https://stats.stackexchange.com/questions/93569/why-is-logistic-regression-a-linear-classifier  
https://stats.stackexchange.com/questions/88603/why-is-logistic-regression-a-linear-model  
http://cs229.stanford.edu/notes/cs229-notes1.pdf



Classical problem solutions (to be done):
- [Imbalanced training data]()
- [Missing data]()
- [Feature selection]()
- [Evaluation matrics]()
- ...

