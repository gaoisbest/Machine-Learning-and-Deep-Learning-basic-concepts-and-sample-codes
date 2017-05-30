# Basic concepts
- **Image classification** is a fundamental task for other Computer Vision problems (such as object detection, segmentation or image captioning).
- In computer view, an image is viewed as a **3-D array**. For a three color channels (RGB) image with 20 pixels wide, 30 pixels tall, it has a total of 20 * 30 * 3 = 1800 integers. Each integer ranges between 0 (black) and 255 (white). The image can be flatten out to be one dimensional array. For example, a specific 32x32 colour image of the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset can be flatten out to a 1x3072 numpy array of uint8s. The first 1024 entries contain the red channel values (row-major order, which means the first 32 entries are the red channel values of the *first row* of the image), the next 1024 the green, and the final 1024 the blue. see [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) for more detailed information.
- Machine learning approach is the **data-driven** approach. And all the available data can be split into training set, validation set and test set. Validation set (Cross validation) is used to select hyper-parameters, and test set is only used to report the accuracy **at a single time** in the end. Note that, when all the hyper-parameters are determined by cross validation, the final model is re-trained on (training set + validation set).
- **Softmax function** maps a vector of arbitrary real-valued scores to a vector of values between zero and one that sum to one. Then, **cross entropy** can be applied. 
- Three approaches for image classification
  - k-Nearest Neighbor classifier, based on the pixel distance (L1 or L2). 
    - `k` and distance measure (L1 or L2) are determined by cross validation. 
    - The drawbacks of kNN are (1) storing all training set and (2) expensive predicting. 
    - Since totally different images may have the same distance, the performance of kNN is not good. [FLANN](http://www.cs.ubc.ca/research/flann/) provides the implementaion of approximate nearest neighbor. 
  - Linear classifier.
    - **Score function** maps raw data to class scores (the score is weighted sum of all pixels); **loss function** quantifies the agreement between the predicted class scores and the ground truth labels.
    - Treat it as a **optimization** problem and the goal is **minimizing the loss**.
    - **f=WX+b**. **W** is **weights** and **b** is **bias** (affect the final score but donot interact with the input **X**. Without **b**, the classifier line will be forced to **cross the origin**). **Bias trick**: combine **W** and **b** into one matrix and extend **x** with constant one.
    - Data preprocessing: zero meaning centering and unit variance.
    - Loss function
      - Multiclass SVM loss: expect the correct class score is at least larger than the incorrect class score by the margin `1`.
      - Cross-entropy loss: unnormalized log probabilities -> softmax (normalized class probabilties) -> minimizing the negative log likelihood of correct class [MLE] (aka minimizing the cross entropy [KL divergence] between the predicted class score and the true label)
      - Final loss: multiclass SVM / cross-entropy + regularization loss. **With regularization loss, the final loss cannot be equal to zero**.
  - Convolutional Neural Network.

# Codes
Vectorized codes are perfered since they are efficient.

## Python List iteration
```
# list iteration
a = ['1', '2', '3', '4', '5']
for idx, ele in enumerate(a):
    print 'index:%d, value:%s' % (idx, ele)
```
## find the most frequent numbers in an array
```
# see https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
a = np.array([1, 1, 3, 7, 7, 7])
np.bincount(a) # array([0, 2, 0, 1, 0, 0, 0, 3], dtype=int64), 3 means 7-index occurs three times 
np.argmax(np.bincount(a)) # 7
```

## L1 and L2 distance between two vectors
```
a = np.array([1,2,3,4])
b = np.array([2,3,4,5])

# two strategies for L1
np.linalg.norm(a-b, ord=1) # 4
np.sum(np.abs(a-b)) # 4

# two strategies for L2
np.linalg.norm(a-b) # 2.0
np.sqrt(np.sum(np.square(a-b))) # 2.0
```
## Euclidean distance between two matrix
```
train = np.random.random((5,3))
test = np.random.random((8,3))

# stratege 1
train_square = np.sum(np.square(train), axis=1) # shape: (5L,)
test_square = np.sum(np.square(test), axis=1, keepdims=True) # shape(8L, 1L)
train_test = np.dot(test, train.T) # (8L, 5L)
res = np.sqrt(-2*train_test + train_square + test_square) # (8L, 5L)

# strategy 2
from scipy.spatial.distance import cdist
res_scipy = cdist(test, train, metric='euclidean')
```

## Image preprocessing and bias trick
```
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0) # X_train shape is N (Number of samples) * D (Dimension)
# subtract the mean
X_train -= mean_image
# third: append the bias dimension of ones (i.e. bias trick)
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
```

## Multi-class SVM loss for one training example (x, y)
```
def L_i_vectorized(x, y, W):
  delta = 1.0
  scores = W.dot(x)
  margins = np.maximum(0, scores - scores[y] + delta)
  margins[y] = 0 # the loss exclude y-th true class itself
  loss_i = np.sum(margins)
  return loss_i
```

## Numerical stability for softmax function
```
f = np.array([123, 456, 789])
p = np.exp(f) / np.sum(np.exp(f)) # Numeric potential blowup

# logC = -max(f)
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f))
```

# Part solution of Assignment 1 

## knn.ipynb

kNN classifier includes two steps: 
First, compute the distances between test images and all train examples. Then, find the k nearest examples and vote for the predicted label.
```
# In k_nearest_neighbor.py
# compute_distances_two_loops computes the distance matrix one element at a time
for i in xrange(num_test):
    for j in xrange(num_train):        
        dists[i, j] = np.linalg.norm(X[i,:] - self.X_train[j,:])\
        # or
        # dists[i,j] = np.sqrt(np.sum(np.square(X[i,:] - self.X_train[j,:])))

# predict_labels
for i in xrange(num_test):
    closest_y = self.y_train[np.argsort(dists[i,])[0:k]]
    y_pred[i] = np.argmax(np.bincount(closest_y))
```

References:  
http://cs231n.github.io/classification/  

http://cs231n.github.io/linear-classify/
