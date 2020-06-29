import glob
import librosa
import os
import numpy as np
from constant import *
import argparse


def audio_clip(data_dir, N, low, high, duration, output_dir):
    speakers = glob.glob(os.path.join(data_dir, "*.sph"))
    speakers.extend(glob.glob(os.path.join(data_dir, "*.wav")))
    for i in range(len(speakers)):
        p = os.path.join(output_dir, str(i))
        if not os.path.exists(p):
            os.makedirs(p)
        y, _ = librosa.load(speakers[i], sr=SAMPLING_RATE)
        for j in range(N):
            k = int(np.random.randint(low, high, size=1))
            librosa.output.write_wav(os.path.join(p, str(j)) + ".wav", 
              y[k*SAMPLING_RATE : (k+duration)*SAMPLING_RATE], SAMPLING_RATE)

parser = argparse.ArgumentParser("The function is to pack the audio files")
parser.add_argument("-d", "--dir", type=str, help="root directory which \
                    contains the fold of audio files from each speaker")
parser.add_argument("-o", "--out", type=str, help="output file name")
args = parser.parse_args()

audio_clip(data_dir=args.dir,output=args.out,N=args.num,low=0,high=600,duration=5)
audio_clip(data_dir="/Users/xuqidiao/Desktop/dc/DeWave/DeWave/data",output_dir='/Users/xuqidiao/Desktop/dc/DeWave/DeWave/data/val',N=128,low=0,high=600,duration=5)