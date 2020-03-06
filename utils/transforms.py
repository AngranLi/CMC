import torch
from torchvision.transforms import functional as F
from torchaudio import transforms
import torch.nn as nn
import numpy as np
import pywt as pywt
import random as rand
from matplotlib import pyplot as plt

from models.threshold import S3


class LowPass:

    def __init__(self, cutoff, dt=1/200, pad_length=None):
        """Low-pass filter.

        Filter is applied in frequency domain before the signal is converted back to the time
        domain. This operation will change the shape of the data (i.e., it is a resampling
        operation).
        
        Arguments:
            cutoff {float} -- Frequency cutoff.
        
        Keyword Arguments:
            dt {float} -- Sampling period. (default: {1/200})
            pad_length {int} -- Length of returned time series. (default: {None})
        """
        self.dt = dt
        self.cutoff = cutoff
        self.pad_length = pad_length

    def __call__(self, x):
        C,n = x.shape

        mu = np.mean(x, axis=1).reshape(C, 1)

        # Apply the fft to the data
        X = np.fft.fft(x - mu, axis=1)

        # Find the freqs above the cut off
        freqs = np.fft.fftfreq(n, self.dt)
        inds = np.abs(freqs) <= self.cutoff

        # Truncate spectrum above cutoff
        X_low = X[:, inds]

        # Pad to signal length
        if self.pad_length:
            n_pad = self.pad_length - X_low.shape[1]
            assert n_pad >= 0, 'Pad Length is shorter than signal length'
            X_low = np.pad(X_low, [(0,0), (0, n_pad)], mode='constant')
        
        x_low = np.fft.ifft(X_low) + mu

        return np.real(x_low)


class ReduceToPoles:
    """Calculate vector norm for each sensor over time."""

    def __call__(self, x):
        C,n = x.shape

        # Pole 1 top and bottom
        p10 = np.linalg.norm(x[:3], axis = 0)
        p11 = np.linalg.norm(x[3:6], axis = 0)

        # Pole 2 top and bottom
        p20 = np.linalg.norm(x[6:9], axis = 0)
        p21 = np.linalg.norm(x[9:], axis = 0)

        xr = np.stack([p10, p11, p20, p21])

        return xr

class Normalize:
    """ Calculate instance norm"""

    def __call__(self,x):
        C,n = x.shape
        x = np.expand_dims(x,axis=0)
        instance_norm = nn.InstanceNorm1d(C)
        normalized_x = np.squeeze(np.array(instance_norm(torch.tensor(x))))
        return normalized_x

class AddSensorRatio:
    """Add signal ratios from different sensors as additional features"""
    
    def __call__(self, x):
        C, n = x.shape
        feats = []
        eps = 1e-10
        for col in range(C):
            indices = np.arange(C)
            feats.append(x[col,:]/(x[indices!=col,:]+eps))
        feats = np.concatenate(feats)
        x_merged = np.concatenate((x,feats))
        return x_merged

class RemapTarget:

    def __init__(self, new_map, old_map):
        """Convert from one class-ID map to another.
        
        Arguments:
            new_map {dict} -- New class-ID mapping.
            old_map {[type]} -- Old class-ID mapping.
        """
        self.new_map = new_map
        self.old_map = old_map

    def __call__(self, old_class_id):

        class_name = self.old_map[old_class_id]
        new_class_id = self.new_map[class_name]

        return new_class_id



class FromNumpy(object):

    def __call__(self, x):
        return torch.from_numpy(x).float()


class Spectrogram(object):

    def __init__(self, n_fft, hop_length):
        """Calculate spectrogram of a set of 1D signals.
        
        The first dimension is batch/channel and the second should be time.
        
        Arguments:
            n_fft {int} -- Size of time window over which to calculate each FFT.
            hop_length {int} -- The stride length between the start of each FFT window.
        """
        self.spec_fn = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    
    def __call__(self, x):
        x = x - x.mean(dim=1).unsqueeze(1)
        x = self.spec_fn(x) + 1e-7
        x = x.log()

        return x


class SwapPillars(object):
    """Randomly flip left and right sensors."""

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if rand.random() < self.probability:
            return np.vstack([x[6:12], x[0:6]])
        else:
            return x


class SwapTopBottom(object):
    """Randomly flip top and bottom sensors."""

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if rand.random() < self.probability:
            return np.vstack([x[3:6], x[0:3], x[9:12], x[6:9]])
        else:
            return x


class Rescale(object):
    
    def __init__(self, upper=6.9, lower=0.1, probability=0.5, preserve_power=True, scale_factor=2):
        """Rescales time series for data augmentation
        
        Keyword Arguments:
            upper {float} -- Clipping upper bound of time series (default: {6.9})
            lower {float} -- Clipping lower bound of time series (default: {0.1})
            probability {float} -- Probability of applying rescale transform (default: {0.5})
            preserve_power {bool} -- Preserve power of time series when rescaling? (default: {True})
            scale_factor {int} -- Range of scaling factors. Goes from 1/s to s (default: {2})
        """
        self.upper = upper
        self.lower = lower
        self.probability = probability
        self.preserve_power = preserve_power
        self.scale_factor = scale_factor

    def __call__(self, x):
        """Apply rescale transform to time series
        
        Arguments:
            x {array} -- Array of time series of size (n_channels, len_of_timeseries)
        
        Returns:
            array -- Rescaled array of same size as input
        """
        if rand.random() < self.probability:
            scaled_list = []
            horiz_factor = rand.uniform(1 / self.scale_factor, self.scale_factor)
            p_orig = 0
            p_scaled = 0
            mean = np.mean(x)
            
            for arr in x:
                arr = self.unclip(arr)
                p_orig += self.power(arr)

                # Rescale time series horizontally
                arr = arr - mean
                n = len(arr)
                scaled = np.interp(np.linspace(0, n, horiz_factor * n), np.arange(n), arr)

                m = len(scaled)

                # Keep middle section if scaled larger than original size
                if m > n:
                    s = int((m - n) / 2)
                    scaled = scaled[s:s+n]
                
                # Interpolate edges using cubic spline if smaller than original size
                elif n > m:
                    diff = n - m
                    flen = int((n - m) / 2)
                    blen = diff - flen

                    
                    t1 = 0
                    t2 = flen
                    n1 = 0
                    n2 = scaled[0]
                    d1 = 0
                    d2 = scaled[1] - scaled[0]
                    A = np.array([
                        [t1 ** 3, t1 ** 2, t1, 1],
                        [t2 ** 3, t2 ** 2, t2, 1],
                        [3 * t1 ** 2, 2 * t1, 1, 0],
                        [3 * t2 ** 2, 2 * t2, 1, 0]
                    ])
                    f = np.array([n1, n2, d1, d2])
                    try:
                        coeffs = np.linalg.solve(A, f)
                    except np.linalg.LinAlgError:
                        coeffs = [0, 0, 0, scaled[0]]

                    front = [
                        coeffs[0] * t ** 3 + coeffs[1] * t ** 2 + coeffs[2] * t + coeffs[3] 
                            for t in range(flen)
                    ]
                    
                    # Cubic spline of right edge
                    t1 = 0
                    t2 = blen
                    n1 = scaled[-1]
                    n2 = 0
                    d1 = scaled[-1] - scaled[-2]
                    d2 = 0
                    A = np.array([
                        [t1 ** 3, t1 ** 2, t1, 1],
                        [t2 ** 3, t2 ** 2, t2, 1],
                        [3 * t1 ** 2, 2 * t1, 1, 0],
                        [3 * t2 ** 2, 2 * t2, 1, 0]
                    ])
                    f = np.array([n1, n2, d1, d2])
                    try:
                        coeffs = np.linalg.solve(A, f)
                    except np.linalg.LinAlgError:
                        coeffs = [0, 0, 0, scaled[-1]]

                    back = [
                        coeffs[0] * t ** 3 + coeffs[1] * t ** 2 + coeffs[2] * t + coeffs[3]
                            for t in range(blen)
                    ]

                    scaled = np.append(front, scaled)
                    scaled = np.append(scaled, back)
                
                scaled_list.append(scaled)
                p_scaled += self.power(scaled)

            scaled = np.vstack(scaled_list)
            
            # Scale vertically to preserve power
            if self.preserve_power:
                vert_factor = np.sqrt(p_orig / p_scaled)
            else:
                vert_factor = rand.uniform(1 / self.scale_factor, self.scale_factor)

            # Clip signal to 0-7
            scaled = np.clip(vert_factor * scaled + mean, self.lower, self.upper)
                
            return scaled
        else:
            return x
        
    def unclip(self, x):
        """Unclip signal using cubic spline to restore information beyond 0-7
        
        Arguments:
            x {array} -- Single time series with single channel. 1D array
        
        Returns:
            array -- Unclipped signal with same shape of original
        """
        if np.max(x) < self.upper-1 and np.min(x) > self.lower+1:
            return x
        t1 = None
        t2 = None
        n1, n2 = None, None
        d1, d2 = None, None
        up, down = False, False
        for i, n in enumerate(x):
            if t1 is None:
                if n >= self.upper - 1 or n <= self.lower + 1:
                    if n >= self.upper - 1:
                        up = True
                    else:
                        down = True
                    t1 = i
                    n1 = n
                    if i == 0:
                        d1 = 0
                    else:
                        d1 = n - x[i - 1]
            
            elif t2 is None:
                if (up and n < self.upper - 1) or (down and n > self.lower + 1):
                    t2 = i
                    n2 = n
                    if i+1 >= len(x):
                        d2 = -d1
                    else:
                        d2 = x[i + 1] - n

                    A = np.array([
                        [t1 ** 3, t1 ** 2, t1, 1],
                        [t2 ** 3, t2 ** 2, t2, 1],
                        [3 * t1 ** 2, 2 * t1, 1, 0],
                        [3 * t2 ** 2, 2 * t2, 1, 0]
                    ])
                    f = np.array([n1, n2, d1, d2])
                    coeffs = np.linalg.solve(A, f)

                    for j in range(t2 - t1):
                        t = j + t1
                        x[t] = coeffs[0] * t ** 3 + coeffs[1] * t ** 2 + coeffs[2] * t + coeffs[3]
                    t1 = None
                    t2 = None
                    n1, n2 = None, None
                    d1, d2 = None, None
                    up, down = False, False
        return x

    def power(self, x):
        """Get power of signal
        
        Arguments:
            x {array} -- single time series signal
        
        Returns:
            float -- power of x
        """
        x = x - np.mean(x)
        return sum(x ** 2) / len(x)


class GetS3Mask(object):
    """Mask data by setting values to zero if S3 is below threshold.

    Arguments:
        threshold {float} -- S3 threshold.
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.s3 = S3()

    def __call__(self, x):
        """Set x to zeros if S3 value is below threshold.

        Arguments:
            x {numpy.ndarray} -- TMS sample of size 12x1000.

        Returns:
            numpy.ndarray -- Masked TMS sample of size 12x1000.
        """

        s3_mask = self.s3(torch.as_tensor(x).unsqueeze(0))[:, 1] < self.threshold

        if s3_mask[0]:
            x = x * 0

        return x


class Denoise(object):

    def __init__(self, gain_range=200, num_avg=2):
        """Removes noise from signals by looking for correlations between sensors pointed in the
        same direction.
        
        Keyword Arguments:
            gain_range {int} -- Number of points at beginning where only noise is present
                (default: {200})
            num_avg {int} -- Number of signals to average (default: {2})
        """
        self.gain_range = gain_range
        self.num_avg = num_avg

    def __call__(self, x):
        # De-mean signal
        x = x - np.expand_dims(np.mean(x, axis=1), axis=1)

        # Find gain ratios between channels
        gains = [np.ptp(x[i][0:self.gain_range]) for i in range(12)]


        # Iterate through x,y,z channel directions
        for i in range(3):
            # Find channels with smallest peak-to-peak
            ptps = [np.ptp(x[i+3*j] / gains[i+3*j]) for j in range(4)]
            idxs = np.argsort(ptps)

            # Get average signal of num_avg smallest channels
            avg = np.zeros(x.shape[1])
            for num, j in enumerate(idxs):
                if num == self.num_avg:
                    break
                avg += x[i+3*j] / (gains[i+3*j] * self.num_avg)
            
            # Remove average from all signals, scaled by gain factor
            for j in range(4):
                x[i+3*j] = x[i+3*j] - avg * gains[i+3*j]
        
        return x

class Crop(object):

    def __init__(self, ratio=(1/5, 3/5)):
        """Crops time dimension to return only fraction of signal
        
        Keyword Arguments:
            ratio {tuple(float)} -- Upper and lower bound of signal to return, expressed as ratios.
                (default: {(1/5, 3/5)})
        """
        self.ratio = ratio

    def __call__(self, x):
        return x[:, int(x.shape[1] * self.ratio[0]):int(x.shape[1] * self.ratio[1])]


class FFT(object):
    """Calculate magnitude of FFT.
    
    FFT is taken for each item in the last dimension of the input.
    """
    def __init__(self, signal_ndim):
        self.signal_ndim = signal_ndim

    def __call__(self, x):
        f = torch.rfft(x, self.signal_ndim, normalized=True)
        f = torch.norm(f, dim=len(f.shape) - 1)

        return f


class CWT1D:
    """Calculate 1D CWT.

    CWT is taken for each item in the last dimension of the input.
    """
    def __init__(self,scales,wavelet):
        self.scales = scales
        self.wavelet = wavelet
    
    def __call__(self,x):
        cwt_matrix, freqs = pywt.cwt(x,self.scales,self.wavelet)
        return cwt_matrix


def trace_zero_pad(x, length):
    """Pad the tensor x with zeros along the trace dimension until is has size equal to length.

    Arguments:
        x {torch.tensor} -- Tensor with dimension (channels x traces x samples).

    Returns:
        torch.tensor -- Tensor with zero padding along trace dimension (at end of trace).
    """
    pad_length = length - x.shape[0]
    if pad_length <= 0:
        return x[:length]
    pad_size = ((0, pad_length), (0, 0), (0, 0))
    x_out = np.pad(x, pad_size, "constant", constant_values=0)
    return x_out


class TraceZeroPad:
    """Class wrapper for trace_zero_pad function."""

    def __init__(self, length):
        self.length = length
    
    def __call__(self, x):
        return trace_zero_pad(x, self.length)


def cropper(x):
    time_dim = x.shape[0]
    if time_dim > 70:
        mid_point = int(time_dim / 2)
        lb = max(0, mid_point - 35 + rand.randint(-15, 15))
        ub = min(mid_point + 35 + rand.randint(-15, 15), time_dim)
        x = x[lb:ub, :, :]
    x = x[:, 50:250, :]
    return x


class RemoveFixedArtifacts:
    """Remove static artefacts from CMR data."""

    def __init__(self, edge_samples=10):
        """Initialize transform.

        Keyword Arguments:
            edge_samples {int} -- Number of starting and ending traces to use in calculating mean
                for removal. (default: {10})
        """
        self.edge_samples = edge_samples

    def __call__(self, x):
        x_mean = np.mean((x[:self.edge_samples] + x[-self.edge_samples:]) / 2, axis=0)
        x = x - x_mean

        return x


class AddTraceDifference:
    """Calculate trace-wise forward difference and add as new channels."""

    def __call__(self, x):
        x_diff = np.zeros_like(x)
        x_diff[:-1] = x[1:] - x[:-1]

        x = np.concatenate((x, x_diff), axis=2)

        return x
