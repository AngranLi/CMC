import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import torch

sys.path.append('models')
from threshold import S3


def visualize(in_path, out_path):

    df = pd.read_csv(in_path)

    # Remove count and remove optical sensor columns
    d = df.iloc[:, 1:-1].to_numpy()
    print(d.shape)

    s3_mdl = S3()

    s3 = s3_mdl(torch.as_tensor(d).permute(1, 0).unsqueeze(0))[0, 1]
    print('S3:', s3.item())

    out_path = f'{out_path}_{s3.item():.4f}'

    # Visualize all signals
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(d[:, :3])
    ax[0, 1].plot(d[:, 3:6])
    ax[1, 0].plot(d[:, 6:9])
    ax[1, 1].plot(d[:, 9:12])
    fig.tight_layout()
    
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(os.path.join(out_path, 'raw.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str)
    parser.add_argument('out_path', type=str, nargs='?', default=None)

    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.splitext(args.in_path)[0]
    print(args.out_path)

    visualize(args.in_path, args.out_path)
