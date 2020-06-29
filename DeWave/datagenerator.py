'''
Class DataGenerator:
    Read in the .pkl data sets generated using datagenerator.py
    and present the batch data for the model
'''
import numpy as np
import librosa
import pickle
from numpy.lib import stride_tricks
import os
from constant import *
import pickle

class DataGenerator(object):
    def __init__(self, pkl_list, batch_size):
        '''pkl_list: .pkl files contaiing the data set'''
        self.ind = 0  # index of current reading position
        self.batch_size = batch_size
        self.samples = []
        self.epoch = 0
        # read in all the .pkl files
        print('pkl_list',pkl_list)
        # with open(pkl_list, 'rb') as f:
            # data = pickle.load(f)
            # print(data)
        # for pkl in pkl_list:
            # print(pkl)
            # self.samples.extend(pickle.load(open(pkl, 'rb')))
        self.samples = pickle.load(open(pkl_list,'rb'))
        self.tot_samp = len(self.samples)
        print(self.tot_samp)
        print('samples')
        np.random.shuffle(self.samples)

    def gen_batch(self):
        # generate a batch of data
        n_begin = self.ind
        n_end = self.ind + self.batch_size
        # ipdb.set_trace()
        if n_end >= self.tot_samp:
            # rewire the index
            self.ind = 0
            n_begin = self.ind
            n_end = self.ind + self.batch_size
            self.epoch += 1
            np.random.shuffle(self.samples)
        self.ind += self.batch_size
        return self.samples[n_begin:n_end]
