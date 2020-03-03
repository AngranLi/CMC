import torch
from torch import nn
from torch.nn import functional as F
from torchaudio import transforms


class Resnet(nn.Module):

    def __init__(
        self, base_model, pretrained, num_classes, in_channels=3, train_all=True, transforms=None
    ):
        """Generalized resnet module. Builds a model based on a passed child resnet class.
        
        Arguments:
            base_model {nn.Module} -- Resnet module (e.g., resnet 18, resnet50, etc.).
            pretrained {bool, str} -- If a Bool is passed, it specifices whether or not to load pretrained,
                weights from ImageNet. Otherwise you may pass a path to a state dict, where it is assumed that
                the state dict contains the same archictecture as the one the instance being created.
            num_classes {int} -- Number of output classes.
        
        Keyword Arguments:
            in_channels {int} -- Number of input channels. (default: {3})
            train_all {bool} -- Whether to train all layers of the network. (default: {True})
            transforms {callable} -- Callable that operates on the input data. (default: {None})
        """
        super().__init__()

        # If a path is given, assume its a state dict and load it AFTER resetting the classifier
        if isinstance(pretrained, bool):
            self.layers = base_model(pretrained=pretrained)
        else:
            self.layers = base_model(pretrained=False)

        if not train_all:
            for param in self.layers.parameters():
                param.requires_grad = False

        if in_channels != 3:
            self.layers.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7,
                stride=2, padding=3, bias=False
            )
        self.layers.fc = nn.Linear(self.layers.fc.in_features, num_classes)
        # If pre-trained is a str, load the state dict
        if isinstance(pretrained, str):
            old_dict = self.state_dict()
            new_dict = torch.load(pretrained)

            # Check sizes of all the layers and if they dont match 
            for name, parameter in new_dict.items():
                if(new_dict[name].size() != old_dict[name].size()):
                    new_dict[name] = old_dict[name]

            self.load_state_dict(new_dict)
        
        self.transforms = transforms
    
    def forward(self, x):
        if self.transforms is not None:
            with torch.no_grad():
                x_t = []
                for x_i in x:
                    x_t.append(self.transforms(x_i))
                x = torch.stack(x_t)

        x = self.layers(x)
        return x


class ResnetThreshold(nn.Module):
    def __init__(self, base_model, params):
        super().__init__()
        self.resnet = base_model(**params)

    def forward(self, x):
        dev = 'cpu' if x.get_device() == -1 else x.get_device()
        s3_mask = x.abs().sum(dim=(1, 2)) < 1e-6

        out_rn = self.resnet(x)
        out = []
        for out_i, m_i in zip(out_rn, s3_mask):
            if m_i:
                out.append(torch.tensor([1e2, -1e2], device=dev))
            else:
                out.append(out_i)
        
        out = torch.stack(out)

        return out

