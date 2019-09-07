import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()


def load_weights(weight_file):
    if weight_file is None:
        return
    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


class KitModel(nn.Module):

    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv3d_1 = self.__conv(3, name='conv3d_1', in_channels=1, out_channels=32, kernel_size=(3, 3, 3),
                                    stride=(1, 1, 1), groups=1, bias=True)
        self.conv3d_2 = self.__conv(3, name='conv3d_2', in_channels=32, out_channels=64, kernel_size=(3, 3, 3),
                                    stride=(1, 1, 1), groups=1, bias=True)
        self.conv3d_3 = self.__conv(3, name='conv3d_3', in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                                    stride=(1, 1, 1), groups=1, bias=True)
        self.conv3d_4 = self.__conv(3, name='conv3d_4', in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                                    stride=(1, 1, 1), groups=1, bias=True)
        self.dense_1 = self.__dense(name='dense_1', in_features=6400, out_features=128, bias=True)
        self.dense_2 = self.__dense(name='dense_2', in_features=128, out_features=2, bias=True)

    def forward(self, x):
        conv3d_1 = self.conv3d_1(x)
        conv3d_1_activation = F.relu(conv3d_1)
        conv3d_2 = self.conv3d_2(conv3d_1_activation)
        conv3d_2_activation = F.relu(conv3d_2)
        max_pooling3d_1 = F.max_pool3d(conv3d_2_activation, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0,
                                       ceil_mode=False)
        conv3d_3 = self.conv3d_3(max_pooling3d_1)
        conv3d_3_activation = F.relu(conv3d_3)
        max_pooling3d_2 = F.max_pool3d(conv3d_3_activation, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0,
                                       ceil_mode=False)
        conv3d_4 = self.conv3d_4(max_pooling3d_2)
        conv3d_4_activation = F.relu(conv3d_4)
        max_pooling3d_3 = F.max_pool3d(conv3d_4_activation, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0,
                                       ceil_mode=False)
        dropout_1 = F.dropout(input=max_pooling3d_3, p=0.25, training=self.training, inplace=True)
        flatten_1 = dropout_1.view(dropout_1.size(0), -1)
        dense_1 = self.dense_1(flatten_1)
        dense_1_activation = F.relu(dense_1)
        dropout_2 = F.dropout(input=dense_1_activation, p=0.5, training=self.training, inplace=True)
        dense_2 = self.dense_2(dropout_2)
        dense_2_activation = F.softmax(dense_2)
        return dense_2_activation

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer
