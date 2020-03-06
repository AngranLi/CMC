import torch
import numpy as np
from datasets import TMSDataset, FromNumpy
import matplotlib.pyplot as plt

from .xt_transforms import LowPass

# Plot the average frequency content of each signal
def plot_freq(x, T = 5):

    dt = T/x.shape[1] 

    x = x - x.mean(axis=1).reshape(x.shape[0], 1)
    D = np.fft.fft(x, axis=1)
    Dm = np.mean(np.abs(D), axis=0)

    plt.subplot(1,3,1)
    freqs = np.fft.fftfreq(x.shape[1], d=dt)
    plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(Dm))
    plt.title('Frequency Content')

    plt.subplot(1,3,2)
    plt.plot(x.transpose())
    plt.title('Raw 12 Channels')

    # Get magnitudes at each sensor array
    s0 = np.linalg.norm(x[:3], axis = 0)
    s1 = np.linalg.norm(x[3:6], axis = 0)
    s2 = np.linalg.norm(x[6:9], axis = 0)
    s3 = np.linalg.norm(x[9:], axis = 0)

    plt.subplot(1,3,3)
    plt.plot(s0)
    plt.plot(s1)
    plt.plot(s2)
    plt.plot(s3)
    plt.legend(['p10', 'p11', 'p20', 'p21'])
    plt.title('')

def get_freq(x, T=5):

    dt = T/x.shape[1] 
    x = x - x.mean(axis=1).reshape(x.shape[0], 1)
    D = np.fft.fft(x, axis=1)
    Dm = np.mean(np.abs(D), axis=0)

    Dm = np.fft.fftshift(Dm)
    freqs = np.fft.fftshift(np.fft.fftfreq(x.shape[1], d=dt))

    return Dm, freqs

# Load the data
root_dir = '/scratch/klensink/data/tms/csv/tms_10_19'
transform = LowPass(10)
train_dataset = TMSDataset(root_dir, transform=transform, split='train')
val_dataset = TMSDataset(root_dir, transform=transform, split='val')

# Loop thru and plot data and frequency content
if True:
# if False 
    N = 1000
    inds = np.random.permutation(np.arange(len(train_dataset)))[:N]
    for i in inds:
        d, class_id = train_dataset[i]
        class_name = train_dataset.target_dict_reverse[class_id]

        plt.figure(figsize=(16,8))
        plot_freq(d)
        plt.title(class_name)
        plt.show()

# Get average freq content for N images
if False:
    N = 1000
    inds = np.random.permutation(np.arange(len(train_dataset)))[:N]

    f_avg, _ = get_freq(train_dataset[0][0])
    for i in range(len(train_dataset)):
        d, _ = train_dataset[i]
        f, freqs = get_freq(d)

        f_avg = f_avg + f
    
    plt.semilogy(freqs, f_avg)
    plt.show()




