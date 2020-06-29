# Single-channel blind source separation
This package decomposes two overlapping speech signals, which are recoded in one channel. 
The method is described in deep clustering paper: https://arxiv.org/abs/1508.04306.
The code is based on [zhr1201/deep-clustering](https://github.com/zhr1201/deep-clustering) from the Github.
and [chaodengusc/DeWave](https://github.com/chaodengusc/DeWave?fbclid=IwAR3517vZdQgwNhc8LUDpbd9Oa2WF5tMfUaZslNPV5lcQH93Ad2QeUxfIVRA)

## Requirements
  * Python3.6
  * tensorflow
  * numpy
  * scikit-learn
  * librosa

## Installation
The python is available on PyPI, and you can install it by typing
`pip install DeWave`
  
## File documentation
  * model.py: Bi-LSTM neural network
  * train.py: train the deep learning network
  * infer.py: separate sources from the mixture
  * utility.py: evaluate the performance of the DNN model
  
## Training your speaker separator


## References
  https://arxiv.org/abs/1508.04306


