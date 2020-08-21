import torch.nn as nn
from .network_utils import *
from .network_bodies import *
import sys
sys.path.append(".")
sys.path.append("..")
# from MinimaxParetoFair import *

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body,use_dropout=False, dropout_p=0.5, feature_dim=None):
        super(VanillaNet, self).__init__()
        #patch to use pretrained networks with no feature dim attribute
        if feature_dim is not None:
            body.feature_dim= feature_dim
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        if use_dropout:
            self.reg=nn.Dropout(p=dropout_p)
        else:
            self.reg =nn.Identity()
        # self.to(Config.DEVICE)

    def forward(self, x):
        # phi = self.body(tensor(x))
        phi = self.body(x)
        phi = self.reg(phi)
        y = self.fc_head(phi)
        return y

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.output_dim = output_dim
        self.body = body
        # self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.output_dim))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob