import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

from cmr_convert import convert_p1m


def visualize(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)

    if in_path.endswith('.p1m'):
        data = convert_p1m(in_path)
    else:
        with open(in_path, 'rb') as f:
            data = pickle.load(f)

    # print(json.dumps(data['metadata'], indent=2))
    data = data['traces']

    # Visualize all traces as an image
    data_im = [[], [], [], []]
    for i, trace in enumerate(data):
        data_im[0].append(trace['S21'])
        data_im[1].append(trace['S22'])
        data_im[2].append(trace['S12'])
        data_im[3].append(trace['S11'])
    data_im = np.array(data_im)
    data_im_f = np.abs(np.fft.rfft(data_im, axis=2))
    print(data_im.shape)
    data_im = data_im[:, :, 50:250]

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(data_im[0])
    ax[0, 1].imshow(data_im[1])
    ax[1, 0].imshow(data_im[2])
    ax[1, 1].imshow(data_im[3])
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, 'raw.png'))

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(data_im_f[0])
    ax[0, 1].imshow(data_im_f[1])
    ax[1, 0].imshow(data_im_f[2])
    ax[1, 1].imshow(data_im_f[3])
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, 'raw_f.png'))

    for i in range(len(data_im)):
        data_im[i] = data_im[i] - np.mean((data_im[i][:10] + data_im[i][-10:]) / 2, axis=0)
    for i in range(len(data_im_f)):
        data_im_f[i] = data_im_f[i] - np.mean((data_im_f[i][:10] + data_im_f[i][-10:]) / 2, axis=0)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(data_im[0])
    ax[0, 1].imshow(data_im[1])
    ax[1, 0].imshow(data_im[2])
    ax[1, 1].imshow(data_im[3])
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, 'cleaned.png'))

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(data_im_f[0])
    ax[0, 1].imshow(data_im_f[1])
    ax[1, 0].imshow(data_im_f[2])
    ax[1, 1].imshow(data_im_f[3])
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, 'cleaned_f.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str)
    parser.add_argument('out_path', type=str, nargs='?', default=None)

    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.splitext(args.in_path)[0]
    print(args.out_path)

    visualize(args.in_path, args.out_path)
