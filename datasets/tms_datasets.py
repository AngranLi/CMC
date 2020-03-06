import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob, re
import pickle

__all__ = ['TARGET_DICT', 'loader_csv', 'loader_binary', 'TMSDataset']


TARGET_DICT = {
    'longgun': 1,
    'remington': 1,
    'shotgun': 1,
    'ar15': 1,
    
    'machete': 1,
    'knife': 1,

    'handgun': 0,
    'beretta': 0,
    'glock': 0,
    'sw357': 0,

    'phone': 0,
    'purse': 0,
    'tablet': 0,
    'unknown': 0,
    'wallet': 0,
    'binder': 0,
    'drill': 0,
    'earbuds': 0,
    'keys': 0,
    'laptop': 0,
}


def loader_csv(filepath, transforms=None):
    """Load TMS data from CSV file format.
    
    Arguments:
        filepath {str} -- Path.
    
    Keyword Arguments:
        transforms {callable} -- Data transformation. (default: {None})
    
    Returns:
        torch.Tensor -- Tensor of size 12x1000.
    """
    # Read the raw CSV
    df = pd.read_csv(filepath)

    # Remove count and remove optical sensor columns
    d = df.iloc[:, 1:-1].to_numpy().transpose()

    # Transform the data
    if transforms:
        d = transforms(d)
    
    return d


def loader_binary(filepath, transforms=None, num_points=1000, num_cols=13):
    """Load TMS data from binary file format.
    
    Arguments:
        filepath {str} -- Path.
    
    Keyword Arguments:
        transforms {callable} -- Data transformation. (default: {None})
    
    Returns:
        torch.Tensor -- Tensor of size 12x1000.
    """
    buffer = np.fromfile(filepath, dtype=float)
    d = buffer.reshape((num_points, num_cols))[:, :(num_cols-1)].transpose()

    # Transform the data
    if transforms:
        d = transforms(d)
    
    return d


class TMSDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        split='train',
        target_dict=None,
        class_map=None,
        loader=loader_csv,
        ext='csv',
        path_include=None,
        path_exclude=None, 
        get_label=True, 
        get_tags=False
    ):
        """Pytorch dataset class for loading TMS time series. 
        Assumes that the data is stored as: root_dir/split/class_name/XX.csv
        
        Arguments:
            root_dir {str} -- Path to the data directory
        
        Keyword Arguments:
            transform {object} -- Transforms that are applied directly to the data.
                (default: {None})
            split {str} -- The data split to load from. Assumes that this is a subdirectory of
                `root_dir`. (default: {'train'})
            target_dict {dict} -- Mapping from file tags to class indices. File tags are the set of
                items identified by splitting a file's parent directory on "_". If None,
                use the default dict defined in the class header. (default: {None})
            class_map {callable} -- Callable that returns resolves a sample's tags to an index. If
                None, will use the max index defined by `target_dict` for all the files tags.
                (default: {None})
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.loader = loader
        self.ext = ext
        self.path_include = path_include
        self.path_exclude = path_exclude
        self.get_label = get_label
        self.get_tags = get_tags

        # Get class to index mapping
        if target_dict is not None:
            self.target_dict = target_dict
        else:
            self.target_dict = TARGET_DICT

        if class_map is not None:
            self.class_map = class_map
        else:
            self.class_map = self._get_class_id

        # Build path list for all files
        if not isinstance(ext, list):
            ext = [ext]
        if not isinstance(root_dir, list):
            root_dir = [root_dir]
        
        self.filepaths = []
        for r in root_dir:
            for e in ext:
                self.filepaths.extend(
                    sorted(glob.glob(os.path.join(r, split, f'**/*.{e}'), recursive=True))
                )

        if self.path_include is not None:
            self.filepaths = [fp for fp in self.filepaths if re.search(self.path_include, fp)]
        if self.path_exclude is not None:
            self.filepaths = [fp for fp in self.filepaths if not re.search(self.path_exclude, fp)]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):

        filepath = self.filepaths[i]
        
        # Read the raw CSV
        if isinstance(self.loader, dict):
            loader_current = self.loader[filepath.split('.')[-1]]
        else:
            loader_current = self.loader
        d = loader_current(filepath)

        if self.transform:
            d1 = self.transform[0](d)
            d2 = self.transform[1](d)
        else:
            raise ValueError('There gotta be some transformations!')

        # Get the classid
        tags = os.path.basename(os.path.dirname(filepath)).split('_')[2:]
        class_id = self.class_map(tags)

        inputs=torch.cat([d1, d2], dim=0)

        out = [inputs]
        if self.get_label:
            out.append(class_id)
        else:
            out.append(d)
        if self.get_tags:
            out.append("_".join(tags))
        
        return tuple(out)
    
    def _get_class_id(self, tags):
        if len(tags) == 0:
            return 0
        return max(self.target_dict.get(t, 0) for t in tags)
