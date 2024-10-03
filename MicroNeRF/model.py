import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import torch
from torch import nn
from torch.nn import functional as F

from torch import nn
from abc import abstractmethod
from utils.pos_enc import encode_position
from typing import List, Callable, Union, Any, TypeVar, Tuple

Tensor = TypeVar('torch.tensor')

class OfficialNerfMean(nn.Module):
    def __init__(self, pos_in_dims, dir_in_dims, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(OfficialNerfMean, self).__init__()
        # pos_in_dims = 76
        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.layers1 = nn.Sequential(
            nn.Linear(D + pos_in_dims, D), nn.ReLU(),  # shortcut
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D // 2), nn.ReLU())
        self.fc_rgb = nn.Linear(D // 2, 1)  # Change to grayscale

        self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_rgb.bias.data = torch.tensor([0.02]).float()

    def forward(self, pos_enc, args, dir_enc=None, z_total=130, z_sample=9, overlap=True, test=False):
        """
        # H,W,3 (x,y,z) => H, W, L => H, W, 1
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """

        # print("pos_enc",pos_enc.shape)
        # B,H,W,N,D
        rgb_mean = []
        B, H, W, D = pos_enc.shape
        # print("In",pos_enc.shape)
        # print(z_sample)
        pos_enc_tmp = pos_enc.repeat(z_sample, 1, 1, 1, 1)

        if z_sample % 2 == 0:
            offset_end = z_sample // 2
        else:
            offset_end = z_sample // 2 + 1
        offset = torch.arange(-(z_sample // 2), offset_end, 1)
        offset = offset * 2.0 / z_total
        offset = offset.repeat(B, H, W, 1)
        offset = offset.permute(3, 0, 1, 2).cuda()

        pos_enc_tmp[:, :, :, :, 2] = pos_enc_tmp[:, :, :, :, 2] + offset
        pos_enc_tmp = encode_position(pos_enc_tmp)  # [6, 17, 128, 128, 512]

        x = self.layers0(pos_enc_tmp)  # (B ,H, W, N_sample, D)
        x = torch.cat([x, pos_enc_tmp], dim=4)  # (B, H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (B, H, W, N_sample, D)

        x = self.rgb_layers(x)  # (H, W, N_sample, D/2)
        rgb = self.fc_rgb(x)  # Output the RGB directly  [6, 17, 128, 128, 1]
        if test:
            return rgb
        rgb_mean = rgb.mean(dim=0).cuda()  # (1,B,H,W,1)
        return rgb_mean, rgb, pos_enc_tmp