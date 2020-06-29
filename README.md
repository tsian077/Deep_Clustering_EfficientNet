# Single-channel blind source separation
This package decomposes two overlapping speech signals, which are recoded in one channel. 
The method is described in deep clustering paper: https://arxiv.org/abs/1508.04306.
The code is based on [zhr1201/deep-clustering](https://github.com/zhr1201/deep-clustering) from the Github.
I fixed issues with inference using the trained model, upgraded the code to
support python3 and made a python package called DeWave.

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
### Prepare training and validation datasets
  1. Put audio files in wav or sph format under the data directory. For each speaker,
     one should create a folder and put all audios that belong to this speaker
     into this folder. The function `dewave-clip` can help generate clips based
     on these audios. As an example, one can download two audio files using the
     links as follows:     
     https://drive.google.com/open?id=1r7FtoEyd_2Xe98OQSs8BciUs7RNry8UW.   
     The source of datasets is from   
     http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus.   
     After downloading the files and put them into the data directory. Under the
     current working directory, create a directory called data. Then type  
     `dewave-clip --dir=data --out=data/train --num=256`  
     in the command line, which will automatically generate the training datasets.
     Similarly, one can type  
     `dewave-clip --dir=data --out=data/val --num=128`  
     to obtain the validation data.
  2. Pack the data. Type  
     `dewave-pack --dir=data/train --out=train.pkl`  
     `dewave-pack --dir=data/val --out=val.pkl`  
     The train.pkl is used as the training data and the val.pkl is used as the
     validation data.
### Train the BI-LSTM
  1. Create two directories. One is used to store trained
     model. The other directory is used to store summary of learning process.
     For example, under the current working directory, we create two directoies,
     namely seeds and summary. Then one can type   
     `dewave-train --model_dir=seeds --summary_dir=summary --train_pkl=train.pkl --val_pkl=val.pkl`  
     in commmand line to start training the BI-LSTM model. Stop the training process once the loss on
     the validation datasets converges.
## Infering based on trained model
  1. For a mixed audio file, e.g. mix.wav, type    
     `dewave-infer --input_file=mix.wav --model_dir=seeds`    
     in command line to restore the sources. Two restored audios called mix_source1.wav and 
     mix_source2.wav are generated. One can download a mixed sample through the 
     link:  
     https://drive.google.com/open?id=1s46w2_9IzVA8LdrnirdI6R8o7etq79-R

## Pretrained model
  I have a pretrained model using TED talks from 5 speakers. One can download
  the model through the link below:  
  https://drive.google.com/open?id=1mSsJYighwgAxLC2AFnRXq1GHBuJhQgiC

## Demo
  Here are a few results based on the pretrained model. The mixed audios are
  synthetic data based on Ted talks. For those speakers who have appeared in
  the training datasets, I sampled different time periods to ensure that there
  is no overlap between training data and test data. The demos can be accessed
  using the links below:  
  http://bit.ly/dewave-demo1   
  http://bit.ly/dewave-demo2  
  http://bit.ly/dewave-demo3  

## References
  https://arxiv.org/abs/1508.04306

## Troubleshooting
  1. Error for reading the audio file using librosa.
     Solution: install ffmpeg.

  2. ValueError: Cannot feed value of shape (X, 100, 129) for Tensor
     'Placeholder_2:0', which has shape '(128, 100, 129)'. 
     Solution: the number of audio clips should be at least 128. 
