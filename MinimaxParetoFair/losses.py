import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# class losses(nn.Module):
#     def __init__(self, reduction='mean',type_loss = 2 ):
#         super(losses, self).__init__()
#         self.reduction=reduction
#         self.type_loss = type_loss
#     def forward(self, inputs, targets):
#
#         # print(targets,inputs)
#
#         if self.type_loss == 1:
#             inputs = torch.nn.Softmax(dim=-1)(inputs)
#             ret = torch.abs(inputs - targets)
#
#         elif self.type_loss == 0:
#             inputs = torch.nn.LogSoftmax(dim=-1)(inputs)
#             ret = -1*(inputs*targets)
#
#         else:
#             inputs = torch.nn.Softmax(dim=-1)(inputs)
#             ret = (inputs - targets) ** 2
#
#         if self.reduction != 'none':
#             ret = torch.mean(ret,-1) if self.reduction == 'mean' else torch.sum(ret,-1)
#
#         return ret
class losses(nn.Module):
    def __init__(self, reduction='mean',type_loss = 2,regression = False ):
        super(losses, self).__init__()
        self.reduction=reduction
        self.type_loss = type_loss
        self.regression = regression
    def forward(self, inputs, targets):

        # print(targets,inputs)

        if self.type_loss == 1:
            if not self.regression:
                inputs = torch.nn.Softmax(dim=-1)(inputs)
            # else:
            #     inputs = torch.nn.Tanh()(inputs)
            ret = torch.abs(inputs - targets)

        elif self.type_loss == 0:
            if not self.regression:
                inputs = torch.nn.LogSoftmax(dim=-1)(inputs)
            # else:
            #     inputs = torch.nn.Tanh()(inputs)
            ret = -1*(inputs*targets)

        else:
            if not self.regression:
                inputs = torch.nn.Softmax(dim=-1)(inputs)
            # else:
            #     inputs = torch.nn.Tanh()(inputs)
            ret = (inputs - targets) ** 2

        if self.reduction != 'none':
            ret = torch.mean(ret,-1) if self.reduction == 'mean' else torch.sum(ret,-1)

        return ret