#!/usr/bin/env python
import argparse
from audiopacker import PackData

parser = argparse.ArgumentParser("The function is to pack the audio files")
parser.add_argument("-d", "--dir", type=str, help="root directory which \
                    contains the fold of audio files from each speaker")
parser.add_argument("-o", "--out", type=str, help="output file name")
args = parser.parse_args()

def packclips():
    gen = PackData(data_dir=args.dir, output=args.out)
    gen.reinit()

packclips()