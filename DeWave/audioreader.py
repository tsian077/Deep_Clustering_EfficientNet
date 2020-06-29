'''
Class audio_reader:
    read in a single audio sample for test
'''
import numpy as np
import librosa
from numpy.lib import stride_tricks
import os
from constant import *


def stft(sig, frameSize, overlapFac=0.75, window=np.hanning):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.array(sig, dtype='float64')
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)

class AudioReader(object):
    '''
        read in a single audio sample
    '''
    def __init__(self, data_dir):
        '''Load in the audio file and transform the signal into
        the formats required by the model'''
        # loading and transformation
        speech_mix, _ = librosa.load(data_dir, SAMPLING_RATE)
        # fix the issue at the begining
        speech_mix = np.concatenate((speech_mix, speech_mix, speech_mix), axis=0)
        speech_mix_spec0 = stft(speech_mix, FRAME_SIZE)[:, :NEFF]
        speech_mix_spec = np.abs(speech_mix_spec0)
        speech_phase = speech_mix_spec0 / speech_mix_spec
        speech_mix_spec = np.maximum(
            speech_mix_spec, np.max(speech_mix_spec) / MIN_AMP)
        speech_mix_spec = 20. * np.log10(speech_mix_spec * AMP_FAC)
        max_mag = np.max(speech_mix_spec)
        speech_VAD = (speech_mix_spec > (max_mag - THRESHOLD)).astype(int)
        speech_mix_spec = (speech_mix_spec - GLOBAL_MEAN) / GLOBAL_STD
        len_spec = speech_mix_spec.shape[0]
        k = 0
        self.ind = 0
        self.samples = []
        # feed the transformed data into a sample list
        while(k + FRAMES_PER_SAMPLE < len_spec):
            phase = speech_phase[k: k + FRAMES_PER_SAMPLE, :]
            sample_mix = speech_mix_spec[k:k + FRAMES_PER_SAMPLE, :]
            VAD = speech_VAD[k:k + FRAMES_PER_SAMPLE, :]
            sample_dict = {'Sample': sample_mix,
                           'VAD': VAD,
                           'Phase': phase}
            self.samples.append(sample_dict)
            k = k + FRAMES_PER_SAMPLE
        # import ipdb; ipdb.set_trace()
        n_left = FRAMES_PER_SAMPLE - len_spec + k
        # store phase for waveform reconstruction
        phase = np.concatenate((speech_phase[k:, :], np.zeros([n_left, NEFF])))
        sample_mix = np.concatenate(
            (speech_mix_spec[k:, :], np.zeros([n_left, NEFF])))
        VAD = np.concatenate((speech_VAD[k:, :], np.zeros([n_left, NEFF])))
        sample_dict = {'Sample': sample_mix,
                       'VAD': VAD,
                       'Phase': phase}
        self.samples.append(sample_dict)
        self.tot_samp = len(self.samples)

    def gen_next(self):
        # ipdb.set_trace()
        begin = self.ind
        if begin >= self.tot_samp:
            return None
        self.ind += 1
        return [self.samples[begin]]
