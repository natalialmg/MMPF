from .network_utils import *
from torchvision import models

class ConvStackBody(nn.Module):
    def __init__(self, in_channels=4, in_w=32, in_h=32,
                 kernels=[3,3,3,3], features=[32,32,32,32], strides=[2,2,2,2],pads=None,gate=F.elu,
                 use_dropout=False, dropout_p=0.5,
                 use_batchnorm=False):
                #NatureConv equivalent is kernels=[8,4,3], features=[32,64,64], strides=[4,2,1]
        super(ConvStackBody, self).__init__()
        if pads is None:
            pads =[0 for _ in range(len(strides))]


        self.gate=gate
        expanded_features = [in_channels]+features
        h_w = (in_w, in_h)
        self.convs = nn.ModuleList()
        self.regs = nn.ModuleList()
        for _ in range(len(kernels)):
            self.convs.append(layer_init(nn.Conv2d(
                                expanded_features[_], expanded_features[_+1],
                                kernel_size=kernels[_], stride=strides[_],padding=pads[_]
            )))
            if use_dropout:
                self.regs.append(nn.Dropout2d(p=dropout_p))
            elif use_batchnorm:
                self.regs.append(nn.BatchNorm2d(expanded_features[_+1]))
            else:
                self.regs.append(nn.Identity())
        for _ in range(len(kernels)):
            h_w = conv_output_shape(h_w, kernel_size=kernels[_], stride=strides[_], pad=pads[_])
        self.feature_dim= h_w[0]*h_w[1]*features[-1]

    def forward(self, x):
        for l, conv in enumerate(self.convs):
            reg = self.regs[l]
            x = self.gate(reg(conv(x)))

        x = x.view(x.size(0), -1)
        return x


# class FCBody(nn.Module):
#     def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
#         super(FCBody, self).__init__()
#         dims = (state_dim,) + hidden_units
#         self.layers = nn.ModuleList(
#             [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
#         self.gate = gate
#         self.feature_dim = dims[-1]
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = self.gate(layer(x))
#         return x

# class FCBody(nn.Module):
#     def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, use_batchnorm=False):
#         super(FCBody, self).__init__()
#         dims = (state_dim,) + hidden_units
#         self.layers = nn.ModuleList()
#         self.regs = nn.ModuleList()
#         for _ in range(len(dims) - 1):
#             self.layers.append(layer_init(nn.Linear(dims[_], dims[_ + 1])))
#             if use_batchnorm:
#                 self.regs.append(nn.BatchNorm1d(dims[_ + 1],momentum=0.5))
#             else:
#                 self.regs.append(nn.Identity())
#         self.gate = gate
#         self.feature_dim = dims[-1]
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             reg = self.regs[i]
#             x = self.gate(reg(layer(x)))
#
#         return x

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x




# class FCResnetLayer(nn.Module):
#     def __init__(self, state_dim, hidden_unit=64, gate=F.relu, residual_depth=None, use_batchnorm=False):
#         super(FCResnetLayer, self).__init__()
#         if residual_depth is None:
#             residual_depth = 1
#         self.gate = gate
#         self.layers = nn.ModuleList()
#         self.regs = nn.ModuleList()
#
#         #first layer
#         self.layers.append(layer_init(nn.Linear(state_dim, hidden_unit)))
#
#         #residual block
#         for _ in range(residual_depth - 1):
#             self.layers.append(layer_init(nn.Linear(hidden_unit,hidden_unit)))
#             if use_batchnorm:
#                 self.regs.append(nn.BatchNorm1d(hidden_unit,momentum=0.5))
#             else:
#                 self.regs.append(nn.Identity())
#
#     def forward(self, x):
#         y = x = self.layers[0](x)
#         for i, layer in enumerate(self.layers[1:]):
#             reg = self.regs[i]
#             y = layer(self.gate(reg(y)))
#         return x + y

class FCResnetLayer(nn.Module):
    def __init__(self, state_dim, hidden_unit=64, gate=F.relu, residual_depth=None):
        super(FCResnetLayer, self).__init__()
        if residual_depth is None:
            residual_depth = 1
        self.gate = gate
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(state_dim, hidden_unit))])
        for _ in range(residual_depth - 1):
            self.layers.extend([layer_init(nn.Linear(hidden_unit, hidden_unit))])

    def forward(self, x):
        y = x = self.layers[0](x)
        for layer in self.layers[1:]:
            y = layer(self.gate(y))
        return x + y

#
# class FCResnetBody(nn.Module):
#     def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, residual_depth=1, use_batchnorm=False):
#         super(FCResnetBody, self).__init__()
#         dims = (state_dim,) + hidden_units
#         self.layers = nn.ModuleList()
#         self.regs = nn.ModuleList()
#         self.gate = gate
#         self.feature_dim = dims[-1]
#         for _ in range(len(dims) - 1):
#             self.layers.append(FCResnetLayer(state_dim=dims[_], hidden_unit=dims[_ + 1],
#                                              gate=gate, residual_depth=residual_depth,
#                                              use_batchnorm=use_batchnorm))
#             if use_batchnorm:
#                 self.regs.append(nn.BatchNorm1d(dims[_ + 1] , momentum=0.5))
#             else:
#                 self.regs.append(nn.Identity())
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = self.gate(self.regs[i](layer(x)))
#         return x

class FCResnetBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, residual_depth=1):
        super(FCResnetBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList(
            [FCResnetLayer(state_dim=dim_in, hidden_unit=dim_out, gate=gate, residual_depth=residual_depth)
             for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x


class MyDenseBody(nn.Module):
    def __init__(self,typenet = 'densenet'):
        super(MyDenseBody, self).__init__()
        if typenet == 'resnet':
            self.body = models.resnet34(pretrained=True)
            self.fc_layer = nn.Linear(512, 64) #512 is resnet34 output
            self.bn_layer = nn.BatchNorm1d(64, momentum=0.5)
            self.feature_dim = 64

        else:
            self.body = models.densenet121(pretrained=True)
            self.feature_dim = 1024
        self.type = typenet

    def forward(self, x):

        if self.type == 'resnet':
            x = self.body.conv1(x)
            x = self.body.bn1(x)
            x = self.body.relu(x)
            x = self.body.maxpool(x)

            x = self.body.layer1(x)
            x = self.body.layer2(x)
            x = self.body.layer3(x)
            x = self.body.layer4(x)

            x = F.relu(x)
            x = self.body.avgpool(x)
            x =  torch.flatten(x, 1)
            x = self.fc_layer(x)
            x = self.bn_layer(x)
            out = F.relu(x)
            # out = torch.flatten(x, 1)
        else:
            out = self.body.features(x)
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)

        return out

    def features(self, x):

        if self.type == 'resnet':
            x = self.body.conv1(x)
            x = self.body.bn1(x)
            x = self.body.relu(x)
            x = self.body.maxpool(x)

            x = self.body.layer1(x)
            x = self.body.layer2(x)
            x = self.body.layer3(x)
            out = self.body.layer4(x)

        else:
            out = self.body.features(x)

        return out


