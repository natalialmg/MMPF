import datetime
import numpy as np
from pathlib import Path
from distutils.util import strtobool


from .torch_utils import *
import torch
import time
import pickle
import os
import sys
sys.path.append(".")
sys.path.append("..")

def list_flatten(l):
    return [item for sublist in l for item in sublist]

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

def compareModelWeights(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def get_weight_dict(df,secret_tag):
    N = len(df)
    weight_dict={}
    for s in df[secret_tag].unique():
        weight_dict[s]=N/(len(df.loc[df[secret_tag]==s]))
    return weight_dict

class TravellingMean:
    def __init__(self):
        self.count = 0
        self._mean= 0

    @property
    def mean(self):
        return self._mean

    def update(self, val):
        self.count+=val.shape[0]
        self._mean += ((np.mean(val)-self._mean)*val.shape[0])/self.count




