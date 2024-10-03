import numpy as np
import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    def __init__(self, pos_in_dims):
        super(LinearEncoder, self).__init__()
        self.embed = nn.Linear(3, pos_in_dims)

    def forward(self, x):
        # x_embedding = 2. * np.pi * self.embed(x)
        # x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        x_embedding = self.embed(x)
        return x_embedding

class PeriodsEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=False):
        super(PeriodsEncoding, self).__init__()
        self.include_input = include_input
        self.periods = 2 ** torch.arange(num_freqs)
    def forward(self, x):
        """
        x: (N, 3) tensor of 3D positions
        """
        shapes = list(x.shape)[:-1]
        x = x.unsqueeze(-1)
        freqs = self.periods.unsqueeze(0).unsqueeze(0).to(x.device)
        sins = torch.sin(x * freqs * np.pi)
        coss = torch.cos(x * freqs * np.pi)

        if self.include_input:
            output = torch.cat([x, sins, coss], dim=-1)
        else:
            output = torch.cat([sins, coss], dim=-1)
        view_shape = shapes + [3 * (2 * self.periods.shape[0] + int(self.include_input))]
        output = output.view(*view_shape)
        return output
