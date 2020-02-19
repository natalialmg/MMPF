import torch
from skimage import io, transform
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import Resize, ToPILImage, ToTensor


class PandasDataset(Dataset):
    """Pandas dataset.    """

    def __init__(self, pd, transform=None):
        """
        Args:
            pd: Pandas dataframe, first column is assumed to be filepath
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pd_frame = pd.copy()
        self.transform = transform

    def __len__(self):
        return len(self.pd_frame)

    def __getitem__(self, idx):
        img_name = self.pd_frame.iloc[idx, 0]
        image = io.imread(img_name)
        labels = np.array(self.pd_frame.iloc[idx, 1:].values).astype('int64')
        sample = {'image': image, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)

        return sample

class SimplePandasDataset(Dataset):
    """Pandas dataset.    """

    def __init__(self, pd, transform=None):
        """
        Args:
            pd: Pandas dataframe, first column is assumed to be data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pd_frame = pd.copy()
        self.transform = transform

    def __len__(self):
        return len(self.pd_frame)

    def __getitem__(self, idx):
        X = self.pd_frame.iloc[idx, 0]
        labels = np.array(self.pd_frame.iloc[idx, 1:].values).astype('int64')
        sample = {'X': X, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)

        return sample

class PandasDatasetNPKW(Dataset):
    """Pandas dataset.    """
    def __init__(self, pd, utility_tag='utility_cat', secret_tag='secret_cat', transform=None):
        """
        Args:
            pd: Pandas dataframe, assumed to contain columns ['filepath']
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.U_torch = torch.Tensor(np.vstack(pd[utility_tag].values).astype('float32')) #stacking because they are saved as list of numpy arrays
        self.S_torch = torch.Tensor(np.vstack(pd[secret_tag].values).astype('float32'))
        self.filepaths = pd['filepath'].values
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        X = torch.Tensor(np.load(self.filepaths[idx]))
        U = self.U_torch[idx]
        S = self.S_torch[idx]
        if self.transform:
            X = self.transform(X)

        return X,U,S

class TablePandasDataset(Dataset):
    """Pandas dataset.    """

    def __init__(self, pd, cov_list, utility_tag='utility_cat', sensitive_tag='sensitive_cat', transform=None):
        """
        Args:
            pd: Pandas dataframe,
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.U_torch = torch.Tensor(np.vstack(pd[utility_tag].values).astype('float32')) #stacking because they are saved as list of numpy arrays
        self.S_torch = torch.Tensor(np.vstack(pd[sensitive_tag].values).astype('float32'))
        self.X_torch = torch.Tensor(pd[cov_list].to_numpy().astype('float32'))
        self.transform = transform

    def __len__(self):
        return self.U_torch.shape[0]

    def __getitem__(self, idx):
        # data = self.pd_torch[idx]
        U = self.U_torch[idx]
        S = self.S_torch[idx]
        X = self.X_torch[idx]
        if self.transform:
            X = self.transform(X)

        return X,U,S

class ImageDataset(Dataset):
    """Pandas dataset.    """

    def __init__(self, pd, utility_tag='utility_cat', secret_tag='secret_cat', transform=None):
        self.U_torch = torch.Tensor(np.vstack(pd[utility_tag].values).astype(
            'float32'))  # stacking because they are saved as list of numpy arrays
        self.S_torch = torch.Tensor(np.vstack(pd[secret_tag].values).astype('float32'))
        self.filepaths = pd['filepath'].values
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        X = io.imread(self.filepaths[idx])
        U = self.U_torch[idx]
        S = self.S_torch[idx]
        if self.transform:
            X = self.transform(X)

        return X, U, S

class ToTensorWrapper(object):
    """Convert sample to Tensors."""

    def __init__(self):
        self.PIL_to_tensor = ToTensor()

    def __call__(self, sample):
        ret_dict={}
        for key,val in  sample.items():
            if key == 'image':
                ret_dict[key] = self.PIL_to_tensor(val)
            else:
                ret_dict[key] = torch.from_numpy(val)
        return ret_dict
        # image, labels = sample['image'], sample['labels']
        # return {'image': self.PIL_to_tensor(image),
        #         'labels': torch.from_numpy(labels)}

class ResizeWrapper(object):
    """
    Rescale image wrapper
    """

    def __init__(self, size, interpolation=2):
        self.Resize = Resize(size, interpolation)

    def __call__(self, sample):
        ret_dict = {}
        for key, val in sample.items():
            if key == 'image':
                ret_dict[key] = self.Resize(val)
            else:
                ret_dict[key] = val
        return ret_dict
        # image, labels = sample['image'], sample['labels']
        # return {'image': self.Resize(image), 'labels': labels}

class ColorizeToPIL(object):
    """Make Grayscale images into fake RGB images
    """

    def __init__(self):
        self.ToPil = ToPILImage()
        return

    def __call__(self, X):
        if len(X.shape) < 3:
            X = np.repeat(np.array(X)[:, :, np.newaxis], 3, axis=-1)
        X = self.ToPil(X)
        return X

class ToOneHotWrapper(object):
    """Adds one-hot version of labels"""

    def __init__(self, n_u, n_s):
        self.n_u = n_u
        self.n_s = n_s

    def __call__(self, sample):
        ret = {}
        for key, val in sample.items():
            ret[key] = val
            if key == 'labels':
                ret['utility_one_hot'] = to_categorical(val[0], self.n_u)
                ret['secret_one_hot'] = to_categorical(val[1], self.n_s)
        return ret

#Keras source code
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
