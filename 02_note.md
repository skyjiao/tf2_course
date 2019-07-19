# Crash Course Deep Learning - Session Two - Low Level Tensorflow and Convolution Neural Network

- **Ellipsis** `...` is used here to indicate a placeholder for the rest of the array dimensions not specified.
Think of it as indicating the full slice [:] for all the dimensions in the gap it is placed, so for a 3d array, a[...,0] is the same as a[:,:,0] and for 4d, a[:,:,:,0], similarly, a[0,...,0] is a[0,:,:,0] (with however many colons in the middle make up the full number of dimensions in the array).

- `newaxis` is used to increase the dimension of the existing array by one more dimension. 
    - convert a 1D array to a row or column vector. 
    ```python
    arr = np.arange(4) # of shape (4,)
    row_vec = arr[np.newaxis, :] # of shape (1, 4)
    col_vec = arr[:, np.newaxis] # of shape (4, 1)
    ```
    - you can use it multiple times to promote the array to higher dimensions. 
    ```python
    arr = np.arange(5*5).reshape(5,5) # of shape (5,5)
    arr_5D = arr[np.newaxis, ..., np.newaxis, np.newaxis] # of shape (1, 5, 5, 1, 1)
    ```
- Use ```partial``` to set default values for a function, pretty handy python function.

## Low level API

- Low Level tensorflow API is like numpy with GPU that can do autodiff.

- tensor is immutable

- ```tf.constant``` creates a tensor

- tensor also has broadcasting. 
```@``` as matrix multiplication = ```tf.matmul```

- In GPU computation with tensorflow, the sum operation uses a reduce mecanism but such operation is not perfectly determinist to gain most computation efficiency, which makes the model not reproducible. But PyTorch does have an option to choose to not use it in exchange for some loss of computation speed. 

- in tensorflow, default float type is 32 bits whereas numpy is 64. 

- tf is strict in types, can't do addition between 32 bits and 64 bits. But we can always force by `tf.cast`. 
- strings are atomic in tf

```RaggedTensor``` a list of tensor of different lengths. 

```SparseTensor``` don't have csr or csc representation tho. Just that. 

```tf.Variables```, where .value returns a tensor and .numpy returns a numpy array.

```tf.device``` let the operation remains on specific devices, CPU or GPU. By default, everything is on GPU except ```tf.data``` operations. 

- tf.feature_column and tf.estimator will probably die.

```tf.summary``` API is for building tensorboard.

```tf.lite``` is for compressing. merge operations that are not necessary. 

```tf.graph_util``` converts into computational graph. 

```tf.saved_model``` includes everything includes the training operations. 

```tf.xla``` some operation optimization on graph. some boost on sequence of small operations for ex. 

```tf.sets``` allows to do union, intersect etc.

```tf.nest``` nest of structure that contains tensors. 

- Most of time, when we code low level API, it's for custom cost function, metric etc. 

in general , we can simply passes losses per instance, and keras will take care to compute the mean or to assign proper weight to each instance;

can't do the mean of precision of each batch.
Keep track of internal metric, must be stateful. Writing metric is as simple as loss function, in condition that you can compute the mean. 
We can't train on precision because it's zero gradient except where is undecriable. 

tensorflow function:
- parallel execution
- fusing operations
- polymorphism : generat different graphs at different input. 


Use tf.function as decorator ```@tf.function```

autograph, go through the python source code, capture for loop, assert etc 
use ```tf.range``` instead of range. 

any state modification 

---
# Data API

When data don't fit in memory, we need a data input schedule to manage data more efficiently.

DataSet, as its name indicates, a sequence of data. 

## Model Parallelism
serving large scale is easy: just spawn a lot of models. 

One gpu per layer - doesn't work - one is working, others are sleeping
vertical split: not great. 

Data parallelism
different machines handle different batches. They just measure the mean of gradient together and share the gradient across all machines and remain synchronised. 

## CNN
filters - edge detection
CNN learns the weights on filters

Visual cortex is at back of your head. Partially connected, only react for certain patterns.

Each neuron is only partially connected to previous layer. Same weight (shared) across sub layers. Whatever patterns detected in one part of image, it will be shared to other part the image too. 

hyperparam:
- size of receptive field, default 3x3. kernel size. 
- padding. Same padding, zero padding. Valid padding (=no padding)
- stride. Distance between two receptive field. 

Input has 4 dimensions: batch, height, width and channels
each feature map 

(kernel_height * kernel_weight * channels_in + 1) * channels_out = nb of parameters. 

(each feature map learns a filter, and upper feature maps learn from previous ones)

gradient ascend on input images to show images that activate most each neuron

### Pooling layers
padding, stride, kernel_size
max_pooling, mean_pooling
max_pooling works best, because we keep only stronger features. 

CNN layers with big strikes without pooling works well too sometimes.

On CNN layers, we utilise strike 1, pooling strike 2. We double filters at each cnn. 

Seperable CNN
Depthwise seperable convolutional layer
One filter for each channel, just learn spatial patterns
then CNN with 1*1 looks for color patterns. Much less computational intensive. Normally we don't use it on input layer. Have a regular CNN. 
Xception. 

Optimizers:
closer momentum to 1, more velocity will be conserved. Close to 0, then can't benefical from velocity. 

Adaptive: it's like normalization the gradients. 

CNN architecture
LeNet - Alexnet (2012)- Inception (Googlenet) (2015) - ResNet (2015) 
the trend is to reduce the nb of parameter but increase depth. Seperate learning color and space patterns. 

The reason why CNN works is that the same pattern found in one region could appear in other regions too. Such pattern repetition is where we should apply CNN. CNN requires stationnary to be applied on sequence data. 

YOLO, replace the last dense layer by a CNN layer, which allows to apply on images of any size. 

Semantic segmentation
we loose a lot of spacial information on a conv net becuase the net gets smaller and smaller, basically each pixel just detects whether is there a cat/dog. 

Instance segmentation
Masked-RCNN (He kaiming)

Anomaly detection
We learn what a normal sample is and we capture how far good is away from target sample. But the difficulity is when we have anormal sample in training data, we kind of think that such sample is normal to build the model. 

Novelty detection/Anomaly detection
One-class-SVM, IsolationForest
https://scikit-learn.org/stable/modules/outlier_detection.html


