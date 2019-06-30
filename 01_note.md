# Crash Course Deep Learning - Session One - Neural Network Fundamentals

by Aurélien Géron on June 20, 2019

---

Researcher in neural science found that cortex connections are structured in layered manner, hence the inspiration to 
computer scientist to develop the multiple layer perception (MLP) model.


Some intuitions behind different activation functions:
- `sigmoid`
    
    Basically, it has near zero gradient at positive/negative values and linear near zero, which makes the model
     *difficult* to converge at extreme values, which is quite counter-intuitive. 
  
- `relu`

    Relu is more robust than sigmoid and it's very computation efficient. It simply put all negative values to zero.
    However, once a neuron outputs negative values through every batch, then the neuron is pratically *dead* that 
    always output zero and won't update anymore. In practice, we often observe 20 to 30% of such *dead* neurons in our networks. 

For dense networks (aka, MLP), more layers, more neurons would make the model to learn more details and tend to overfit on training data. And for the sake of tuning simplicity, we can keep the number of neurons equally across layers since the model learn itself to *desactivate" unused neurons at higher layers. We just need to tune the number of neurons per layer and the number of layers. 

 > Large neural networks almost never get stuck in local minima, and even when they do these local optima are almost as good as the global optimum. However, they can still get stuck on long plateaus for a long time.


Learning rate is *the* most important parameter to tune for *any* type of neural networks. Any change to the network should come along with a learning rate retuning. Some thought using adam optimizer, we don't need to handle learning rate, it's wrong. adam simple optimize the relative step size, but you still need to find the optimal learning rate for your problem. I should write a simple learning rate tuning algorithm to quickly find the optimal rate within one epoch. Technically, one simply needs to 



### to do: write my own learning rate optimizer. Check population based tuning by deep mind.

Batch size and learning rate are quite related. If one increases batch size, then one should proportionally increase the learning rate. And François Chollet suggest to always use the largest batch size possible of your GPU when training. 

## Question how to determine the maximum batch size for a given hardware? 

multitask training: we can try to predict many things at the same time and it generally gives better performance than monotask. 

## Question: is multitask always better? Is a neural network just trying to build features from data and performing a linear prediction at the last layer? If we try to learn many things at a time, would there be some counterversal effect on weights? 

*one-shot learning* and *multi-shot learning* would usually reflect to siamese network. 

## Question: when we train siamese network, do we have unbalanced data problem? 

For a regression problem, if we want the result to be positive, instead of using `linear` activation, we could use `softplus`, which is *softer* than `relu`. 

When we use `mse loss`, it's important to clean up out-liners. Or we can use `huber loss` which is basically quadratic near zero but linear at long distance thus more robust to out-liners. 

About random seed problem, tensorflow uses random seed from `tf.random.seed`, but `numpy` has its own random seed from `np.random.seed`. If you use GPU, then it's impossible to make the result totally reproducible because tensorflow optimizes the GPU computation at runtime. Always, *pytorch* does give us an option to choose whether or not we need reproducible result by sacrificing some computation efficiency. 

One can normalize the data to extreme small values that the system's float computation will do some sort of relu for you, thus you don't even need to put an activation function. (# need to check on that)

---

Wide and Deep Network

One can choose features to be send directly to the output layer, especially when they are high correlated with the target variable and that we know they are very predictive. 

Keras's function API allows us to create very flexible network structure. 

    
    model as layers, layers as function
    
---

## Techniques for training deep net

It took us a long time to figure out how to train a deep neural net. 

### 1. Weight Initialization

In order to preserve the weight distribution during training across different layers. And each activation function has its own *optimal* weight initialization function. That's mainly the reason why we can't go deep, because of the vanishing gradient problem. Because, people might thought that we got stuck in local minima.

- Xavier (Glorot) init is good for `sigmoid` or `tanh` activation. It's basically a random gaussian init. 
- He init is good for `relu`. `kernel_initialzer = "he_normal"`

`elu` where e stands for *exponential*, has a smoother part at negative and at zero than `relu` .It should be better than `relu` because the output can be tuned to zero mean however `relu` always output positives. 

`selu` is the state-of-the-art activation function for MLP networks. With `lecun_norm` weight initialization, mathematically it preserves the weight distribution during training across layers, aka *self normalization neural networks*. Always standardize input when using it. 

### 2. Batch Norm

When using it, it's important to shuffle instances after each epoch, otherwise we always gonna use the same batch over epoches. 

During training, since we send data per batch, it's easy to do batch norm, but how to do it on test set if data comes along. So we need to build the batch mean and std during training. Practically, we use exponentially moving average calculated on every batches during training and apply that on test. 

It seems that the reason why batch norm works is totally different than that given by its inventer, aka covariate shift. 

    
    Batch normalization enables the use of higher learning rates, greatly accelerating the learning process.
    
    In practice, restricting the activations of each layer to be strictly 0 mean and unit variance can limit the expressive power of the network. Therefore, in practice, batch normalization allows the network to learn parameters \gamma  and \beta  that can convert the mean and variance to any value that the network desires. 
    
An explanation of batch norm can be found [here](http://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/).

## Question: how to check batch norm's gamma and beta in previous exo. I didn't see anything in the summary. 

### 3. Dropout 
First proposed by Geoffrey Hinton. 

Turn off random neurons at each batch (aka iteration). It forces every neuron to be useful and make the problem harder during training. So it could be that even test result is better than training because all neurons are activated during test. It's similar to the principe of random forest, because we use completely different neural networks during training and kind of ensemble them together at test. 

Another analogy is how *resnet* works with *xgboost*. When build a residual collection, the neuron actually try to predict `f(x) - x` thus the residual, which is just like `xgboost`. 

## Should we put a dropout to each layer? 
---

## Unsupervised pretraining

With giving any image labels, one can *invent* labels (objectives) to train a neural network. For example, we can predict relative position of different regions in an image, we can predict the color of an image by using its uncolored layer, we can (# to check, don't remember).

We can use autoencoder on unlabeled data, then drop decoder and use the encoder to do classification. Such technique is quite useful when we have a huge unlabeled data but only a small part of it is labeled. 



    
    
    
    
