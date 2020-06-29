import argparse
from .train import train

parser = argparse.ArgumentParser("The function is to train the model")
parser.add_argument("-m", "--model_dir", type=str, help="the directory \
                    where the trained model is stored")
parser.add_argument("-s", "--summary_dir", type=str, help="the directory \
                    where the summary is stored")
parser.add_argument("--train_pkl", type=str, nargs='+', \
                    help="file name of the training data")
parser.add_argument("--val_pkl", type=str, nargs='+', \
                    help="file name of the validation data")
args = parser.parse_args()

def trainmodel():
    train(args.model_dir, args.summary_dir, args.train_pkl, args.val_pkl)
