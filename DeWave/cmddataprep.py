#!/usr/bin/env python

## Script is used to generate training and validation datasets from sph files
from .clip import audio_clip
import argparse
parser = argparse.ArgumentParser("generate training data")
parser.add_argument("--dir", type=str, help="input folder name",
default="data")
parser.add_argument("--num", type=int, help="The number of clips for\
each speaker", default=128)
parser.add_argument("--duration", type=int, help="The duration of each \
clip", default=5)
parser.add_argument("--low", type=int, help="starting time of the audio from \
which the clip is sampled", default=0)
parser.add_argument("--high", type=int, help="ending time of the audio from \
which the clip is sampled", default=600)
parser.add_argument("--out", type=str, help="the output directory",
default="data/train")
args = parser.parse_args()

def audioclips():
  audio_clip(args.dir, args.num, args.low, args.high, args.duration, args.out)
