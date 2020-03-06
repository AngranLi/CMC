import torch
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
from torchvision import models, transforms
import numpy as np
from xt_training import metrics

from utils import transforms as xt_transforms
from datasets.tms_datasets import TMSDataset, loader_binary, loader_csv
from models.resnet_1d import resnet18, resnet50
from models.resnet_2d import Resnet, ResnetThreshold


# Transforms
transforms_q = transforms.Compose([
    xt_transforms.GetS3Mask(0.0),
    # xt_transforms.LowPass(13, pad_length=192),
    xt_transforms.SwapPillars(),
    # xt_transforms.ReduceToPoles(),
    # xt_transforms.AddSensorRatio(),
    xt_transforms.Normalize(),
    xt_transforms.FromNumpy()
])
transforms_k = transforms.Compose([
    xt_transforms.GetS3Mask(0.0),
    # xt_transforms.LowPass(13, pad_length=192),
    # xt_transforms.SwapPillars(),
    # xt_transforms.ReduceToPoles(),
    # xt_transforms.AddSensorRatio(),
    xt_transforms.Normalize(),
    xt_transforms.FromNumpy()
])

# Dataloader
batch_size = 256
workers = 4

# Dataset
target_dict = {
    'longgun': 1,
    'remington': 1,
    'shotgun': 1,
    'ar15': 1,
    
    'machete': 1,
    'knife': 1,

    'handgun': 1,
    'beretta': 1,
    'glock': 1,
    'sw357': 1,
}
dataset = {
    'cls': TMSDataset,
    'params': {
        'root_dir': [
            # '/nasty/scratch/common/tms/cramer/gen0',
            '/nasty/scratch/common/tms/kelowna/gen0',
            '/nasty/scratch/common/tms/NYC/gen0',
            '/nasty/scratch/common/tms/reds/gen0',
            '/nasty/scratch/common/tms/UND_multRemoved/gen0',
            '/nasty/scratch/common/cmr-tms/kelowna/gen2/tms/',
            '/nasty/scratch/common/cmr-tms/kelowna/sidebyside/tms',
            '/nasty/scratch/common/cmr-tms/kelowna/unified/tms/',
        ],
        'ext': ['csv', 'bin'],
        'loader': {'csv': loader_csv, 'bin': loader_binary},
        'split': '*',
        'path_exclude': '/test/|/blind-test/|/calibration/',
        'target_dict': target_dict
    }
}
train_split = 0.8
split_objects = True

test_datasets = {
    'gen 1.1' : {
        'cls': TMSDataset,
        'params': {
            'root_dir': '/nasty/scratch/common/tms/kelowna/gen1.1/tms',
            'split': 'calibration',
            'ext': 'csv',
            # 'loader': loader_binary,
            # 'path_include':'test',
            'target_dict': target_dict,
        }
    }
}

# Model
model = {
    'cls': ResnetThreshold,
    'params': {
        'base_model': resnet50,
        'params': {
            'num_classes': 128,
            'in_channels': 12
        }
    }
}
use_best = True

# Loss and metrics
criterion = nn.CrossEntropyLoss()
batch_metrics = {
    'acc30': metrics.Accuracy(0.3),
    'acc50': metrics.Accuracy(0.5),
    'acc70': metrics.Accuracy(0.7),
    'auc': metrics.ROC_AUC()
}

# Optimizer
optimizer = {
    'cls': optim.Adam,
    'params': {
        'lr': 1e-2,
    }
}

# Scheduler
# lr = 'lr'         if epoch < 10
# lr = 0.1 * 'lr'   if 10 <= epoch < 20
# lr = 0.01 * 'lr'  if epoch >= 20
epochs = 30
scheduler = {
    'cls': lr_scheduler.MultiStepLR,
    'params': {
        'gamma': 0.1,
        'milestones': [10, 20]
    }
}
