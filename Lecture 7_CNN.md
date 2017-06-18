
CNN for text classification
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

vocab_size – The size of our vocabulary. This is needed to define the size of our embedding layer, which will have shape [vocabulary_size, embedding_size].

we enable dropout only during training. We disable it when evaluating the model (more on that later).

The first layer we define is the embedding layer, which maps vocabulary word indices into low-dimensional vector representations. It’s essentially a lookup table that we learn from data.

tf.name_scope creates a new Name Scope with the name “embedding”. The scope adds all operations into a top-level node called “embedding” so that you get a nice hierarchy when visualizing your network in TensorBoard.

tf.nn.embedding_lookup creates the actual embedding operation. 

The result of embedding operation is a 3-dimensional tensor of shape [None, sequence_length, embedding_size].

tf.nn.embedding_lookup
tf.name_scope



CNN

A ConvNet architecture is composed of distinct types of Layers (e.g. CONV/FC/RELU/POOL)
Each Layer accepts an input 3D volume and transforms it to an output 3D volume through a differentiable function.
CONV/FC have parameters (the parameters will be trained with gradient descent), and RELU/POOL don’t have parameters.
CONV/FC/POOL have hyperparameters, and RELU doesn’t have hyperparameters.
POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [width/2 x height/2 x depth].
FC will compute the class scores, resulting in volume of size [1 x 1 x number of class]. 


The CONV layer’s parameters consist of a set of learnable filters. 
Every filter is small spatially (along width and height), but extends through the full depth of the input volume. 
For example, a typical filter on a first layer of a ConvNet might have size 5x5x3 (i.e. 5 pixels width and height, and 3 because images have depth 3, the color channels). 
As we slide the filter over the width and height of the input volume we will produce a 2-dimensional activation map that gives the responses of that filter at every spatial position. 

Now, we will have an entire set of filters in each CONV layer (e.g. 12 filters), and each of them will produce a separate 2-dimensional activation map. We will stack these activation maps along the depth dimension and produce the output volume.


CONV layer:
Local Connectivity.
we will connect each neuron to only a local region of the input volume.
The spatial extent of this connectivity is a hyperparameter called the receptive field of the neuron (equivalently this is the filter size).
The extent of the connectivity along the depth axis is always equal to the depth of the input volume. 
The connections are local in space (along width and height), but always full along the entire depth of the input volume.

About the size of the output volume: the depth, stride and zero-padding. We discuss these next:
First, the depth of the output volume is a hyperparameter: it corresponds to the number of filters we would like to use.
Second, we must specify the stride with which we slide the filter. 
The size of this zero-padding is a hyperparameter, which allow us to control the spatial size of the output volumes.

the spatial size of the output volume as a function of the input volume size (W), the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S), 
and the amount of zero padding used (P) on the border. 
correct formula for calculating how many neurons “fit” is given by (W−F+2P)/S+1. 

In general, setting zero padding to be P=(F−1)/2 when the stride is S=1 ensures that the input volume and output volume will have the same size spatially.


Parameter Sharing.
Parameter sharing scheme is used in Convolutional Layers to control the number of parameters.
we are going to constrain the neurons in each depth slice to use the same weights and bias.
During backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights per depth.

To summarize, the Conv Layer:
Accepts a volume of size W1×H1×D1
Requires four hyperparameters:
Number of filters K,
their spatial extent F,
the stride S,
the amount of zero padding P.

Produces a volume of size W2×H2×D2 where:
W2=(W1−F+2P)/S+1
H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
D2=K

With parameter sharing, it introduces F*F*D1 weights per filter, for a total of (F*F*D1)*K weights and K biases.

A common setting of the hyperparameters is F=3,S=1,P=1. 

The backward pass for a convolution operation (for both the data and the weights) is also a convolution (but with spatially-flipped filters). 


Pooling layer. 
Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. 

The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. 

The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. 

The depth dimension remains unchanged. 

it is not common to use zero-padding for Pooling layers


It is worth noting that the only difference between FC and CONV layers is that the neurons in the CONV layer are connected only to a local region in the input, and that many of the neurons in a CONV volume share parameters. 
However, the neurons in both layers still compute dot products, so their functional form is identical. 
