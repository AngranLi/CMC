import torch
from torch import nn
import numpy as np

class S3(nn.Module):
    """S3 threshold model."""

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Linear(1, 1)

    def forward(self, x):
        dev = 'cpu' if x.get_device() == -1 else x.get_device()
        x = x.cpu().numpy()
        out = []
        for x_i in x:
            # Get peak-to-peak for each channel and sensor norms
            p2p = np.array([x_i[i].max() - x_i[i].min() for i in range(12)])
            norms = [
                np.linalg.norm(p2p[:3]),
                np.linalg.norm(p2p[3:6]),
                np.linalg.norm(p2p[6:9]),
                np.linalg.norm(p2p[9:12])
            ]

            # Calculate S^3
            SP1 = np.mean(norms[:2])
            SP2 = np.mean(norms[2:4])
            S3 = (SP1 * SP2 * min(SP1, SP2)) ** (1 / 3)
            
            out.append(torch.tensor([-S3, S3]))
        
        out = torch.stack(out).to(dev)

        return out
