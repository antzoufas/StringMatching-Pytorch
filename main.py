import argparse
import glob
import pandas as pd
import pickle
from utils import load_dataset, run_model
from model import load_model
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

# Define arguments to be passed by the user #
parser=argparse.ArgumentParser()
parser.add_argument('--weights', help='Directory of weights file')
parser.add_argument('--dataset_train', help='What dataset to train on')
parser.add_argument('--dataset_test', help='What dataset to test on')
parser.add_argument('--hidden_units', help='Number of hidden units')
parser.add_argument('--batch_size', help='Batch size')
parser.add_argument('--bidirectional', help='Whether to use bi-RNNs')
parser.add_argument('--highway', help='Whether to use highway network')
parser.add_argument('--self_attention', help='Whether to use self attention')
parser.add_argument('--max_pooling', help='Whether to use max pooling')
parser.add_argument('--alignment', help='Whether to use alignment attention')
parser.add_argument('--shortcut', help='Whether to use shortcut connections')
args=parser.parse_args()
#############################################

# Default parameters #

weights = args.weights
hidden_units = int(args.hidden_units) if args.hidden_units else 60
batch_size = int(args.batch_size) if args.batch_size else 32
bidirectional = bool(args.bidirectional) if args.bidirectional else True
highway = bool(args.highway) if args.highway else False
self_attention = bool(args.self_attention) if args.self_attention else False
max_pooling = bool(args.max_pooling) if args.max_pooling else False
alignment = bool(args.alignment) if args.alignment else False
shortcut = bool(shortcut) if args.shortcut else False

######################

# Load character list #
with open("characters.pkl", 'rb') as char:
    characters = pickle.load(char)
#########################

model = load_model(hidden_units, bidirectional, highway, self_attention, max_pooling, alignment, shortcut)

two_fold = args.dataset_train == args.dataset_test

print("Loading train dataset...")   
dataset_train = load_dataset(args.dataset_train, characters, two_fold, False)
print("Loading test dataset...")
dataset_test = load_dataset(args.dataset_test, characters, False, two_fold)

run_model(model, weights, batch_size, dataset_train, dataset_test, two_fold)
