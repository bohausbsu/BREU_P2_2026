import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AnomalyTransformer import AnomalyAttention
from AnomalyTransformer import train
from AnomalyTransformer.dataset import load_csv, split_data, normalize_splits, make_loaders


def run_train():
    args = train.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, feature_names = train.load_csv(args.csv, skip_cols=0, has_header=True)
    print(f"Loaded {args.csv}")



if __name__ == "__main__":
    pass    