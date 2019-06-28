# RNN

Since one year and half, RNN is out of date already. But it was state of art one year and half ago. 

RNN is still good in time series forecasting
RNN is out of date for NLP now, everything is about transformer today. 

DeepRNN is a stack of memory cells. 

To train, unrolled through time - back propagation through time. 

RNN not stable, Deep RL is much unstable

Seq2Seq, Seq2Vec, Vec2Seq (auto-caption)
Vec2Seq, use the output of CNN as vector to feed into RNN. We train the while end to end structure. 

Encoder-Decoder (seq2seq') allows size difference between input and output. And we need the whole input information to make a translation for example. 

Only works for short term patterns. 

In forecasting, it's not always better to always have more data, window could be chosen pretty short. 

RNN shines when there are high signal noise ratio. 

### what's the diff between RNN and SimpleRNN and SimpleRNNCell

RNN we tend to use tanh to avoid exposing gradient, because we keep feeding back weight to cell itself and the error may get amplified. 
RNN use learning rate warm, start with small learning rate. 

diff with tf1, tf2
tf1 is purely symbolique, graph mode. placeholder from tf1 is disappeared. 
before running the session, nothing will be computed. 

TimeDistributed layer in keras, can wrapper any other layers. TO check

Forecasting the full shifted series
write custom loss function to let different time get different weight. 

phased lstm - learn to activate neurons - when input are not homogenous. 

Add extra features into rnn, add to last output layer by concatenating.

Simple RNN can memorise around 20 time stamps. LSTM can about 100 steps. 

Cell vs Layer:
A layer is composed of multiple cells. 
This allows you to build custom cells. 


Conv1D to do sequence
Could be used as a preprocess layer, before feeding into a RNN. 

Can be used with large strike to shrink data when working with for example daily and weekly data at the same time. 
Use multiple kernels as preprocessing layer. 

WaveNet: CNN without RNN, only conv1D layers. Works well.  Expecially when yuo have long term pattern. State of art text to speech model. 


preivously stateless RNN
In general it works well because value usaully depends on pretty short term memory.
batches are shuffled.
Stateful RNN: The batch needs to be consecutive. Reset the state when you finish one run. 
but the instances are not IID.
Need to well prepare the data. (Stateful RNN is not easy, just prepare the dataset is pretty hard)

### Word Embedding
embedding layer is like onehot encoding followed by a dense layer, which is simply look up operation. 

If we train on hyperpolic space (not euclidien), we may find hierarchical structure of words.
hyperpolic space is like a tree structure. 

recurrent dropout for lstm/rnn layer

use tensorboard to try to visualize embedding

## Question: Embedding is trained on a supervised setup? 

Use tensorflow hub to reuse pretrained embedding.
ELMo
Word2Vec

### Bidirecitonal RNN
Two rnn, and concatenate together
input get duplicated and output get concatenated.
for example, translation uses brnn as encoder.
samplesoftmax speeds up softmax for translation because too many words to compute exp.

## Question: People say that we can use embedding for encode categorical variable? 
### Attention model
Attention or Memory? 
We're not ignoring the rest when we pay "attention" based on the model. So we may need to call it memory. 

Attention is all you need.
512 embedding space, 500 sentence length
stack multiple encoding cells, then sent output to every decoding cells. 
attention is a differentiable lookup

multihead

-> bert and GBT2
We can drop RNN entirely and just use attention. 

BERT trained on prediction by recovering some words missing in sentences. Or two sentences as input, predict whether one follows another. 
Bert and GPT only take one part of the original Attention, either encoder or decoder part. 

take a pretrained bert model, fine tune it on your own dataset, add a last token as target to predict. 
Bert and GPT use subword embedding. 




