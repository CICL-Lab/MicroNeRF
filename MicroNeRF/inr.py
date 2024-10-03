import numpy as np
import math

import torch
from torch import nn
import torch.nn.parallel
import torch.nn.functional as F

from rff.layers import GaussianEncoding

class PSFConvolution3D(nn.Module):
    def __init__(self, psf_kernel=None):
        super(PSFConvolution3D, self).__init__()
        # TODO: add scalable parameter
        if psf_kernel is None:
            psf_kernel = np.array([[[0.0, 0.0, 0.0],
                                       [0.0, 0.1, 0.0],
                                       [0.0, 0.0, 0.0]],

                                      [[0.0, 0.1, 0.0],
                                       [0.1, 0.6, 0.1],
                                       [0.0, 0.1, 0.0]],

                                      [[0.0, 0.0, 0.0],
                                       [0.0, 0.1, 0.0],
                                       [0.0, 0.0, 0.0]]], dtype=np.float32)
        psf_kernel = torch.from_numpy(psf_kernel)
        self.register_buffer('psf_kernel', psf_kernel)

    def forward(self, x):
        return torch.mul(x, self.psf_kernel)

class LearnableAttenuation(nn.Module):
    def __init__(self, initial_alpha=0.1, initial_beta=0.1):
        super(LearnableAttenuation, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.beta = nn.Parameter(torch.tensor(initial_beta))

    def forward(self, depth, pred_rgb):
        depth = (depth + 1) / 2
        pred_rgb = pred_rgb * torch.exp(-self.alpha * depth + self.beta)

        return pred_rgb

class LearnableGaussianConv3D(nn.Module):
    def __init__(self, channels, kernel_size, init_mean=0.0, init_var=1.0):
        super(LearnableGaussianConv3D, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels

        # Learnable parameters
        self.mean = nn.Parameter(torch.tensor(init_mean))
        self.var = nn.Parameter(torch.tensor(init_var))

    def create_gaussian_kernel(self, device):
        # Create coordinate grid
        coords = torch.arange(self.kernel_size, dtype=torch.float32)
        coords = coords - (self.kernel_size - 1) / 2.0
        x, y, z = torch.meshgrid(coords, coords, coords)
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        # Compute Gaussian
        kernel = torch.exp(-((x - self.mean) ** 2 + (y - self.mean) ** 2 + (z - self.mean) ** 2) / (2 * self.var ** 2))

        # Normalize kernel
        kernel = kernel / kernel.sum()

        # Expand to match input channels
        kernel = kernel.expand(self.channels, 1, *kernel.shape).to(device)
        return kernel

    def forward(self, x):
        # Create Gaussian kernel
        raw_shapes = list(x.shape)
        x = x.unsqueeze(-4)
        target_shapes = [-1, 1] + raw_shapes[-3:]
        x = x.view(*target_shapes)
        kernel = self.create_gaussian_kernel(x.device)
        x = F.conv3d(x, kernel, padding=self.kernel_size // 2, groups=self.channels)
        x = x.view(*raw_shapes)
        # Perform convolution
        return x


class LearnablePSF(nn.Module):
    def __init__(self, kernel_size=5):
        super(LearnablePSF, self).__init__()
        self.kernel_size = kernel_size
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Learnable parameter

    def forward(self, raw_img, micro_x=1, micro_y=1, micro_z=1):
        # Create the 3D grid
        half_kernel = self.kernel_size // 2
        raw_shapes = list(raw_img.shape)
        raw_img = raw_img.unsqueeze(-4)
        target_shapes = [-1, 1] + raw_shapes[-3:]
        raw_img = raw_img.view(*target_shapes)
        x, y, z = torch.meshgrid(torch.linspace(-half_kernel*micro_x, half_kernel * micro_x, self.kernel_size),
                                 torch.linspace(-half_kernel * micro_y,  -half_kernel * micro_y, self.kernel_size),
                                 torch.linspace(-half_kernel * micro_z, -half_kernel * micro_z, self.kernel_size),
                                 indexing='ij')

        # Center the grid
        z = z.to(raw_img.device)
        y = y.to(raw_img.device)
        x = x.to(raw_img.device)

        # Calculate r^2
        r_squared = x ** 2 + y ** 2  # This should be revised for convolution integration.

        # Calculate sigma^2
        sigma_squared = self.alpha * torch.abs(z.float())

        # Avoid division by zero
        sigma_squared = torch.clamp(sigma_squared, min=1e-6)

        # Calculate the PSF
        psf = 1 / (2 * math.pi * sigma_squared**2) * torch.exp(-r_squared / (2 * sigma_squared**2))

        # Normalize the PSF
        psf = psf / psf.sum()

        # Reshape the PSF for 3D convolution (1, 1, kernel_size, kernel_size, kernel_size)
        psf = psf.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size)

        # Move PSF to the same device as the input
        psf = psf.to(raw_img.device)

        raw_img = F.conv3d(raw_img, psf, padding=self.kernel_size // 2)
        raw_img = raw_img.view(*raw_shapes)

        # Apply 3D convolution
        return raw_img

class OfficialNerfMean(nn.Module):
    def __init__(self, pos_in_dims, dir_in_dims, D, is_train=True):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(OfficialNerfMean, self).__init__()
        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims
        self.is_train = is_train

        self.enc = GaussianEncoding(sigma=1, input_size=3, encoded_size=int(pos_in_dims/2))

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

        self.fc_feature = nn.Linear(D, D)
        self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D // 2), nn.ReLU())
        self.fc_rgb = nn.Linear(D // 2, 1)  # Change to grayscale

        self.fc_rgb.bias.data = torch.tensor([0.02]).float()

        self.psf_conv = LearnablePSF(kernel_size=5)
        # self.psf_conv = LearnableGaussianConv3D(channels=1, kernel_size=3)
        # self.psf_conv = PSFConvolution3D()

        self.attenuate = LearnableAttenuation()

    def forward(self, coors):
        """
        # H,W,3 (x,y,z) => H, W, L => H, W, 1
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        pos_enc_tmp = self.enc(coors)

        x = self.layers0(pos_enc_tmp.float())  # (B ,H, W, N_sample, D)
        x = torch.cat([x, pos_enc_tmp], dim=-1)  # (B, H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (B, H, W, N_sample, D)

        x = self.rgb_layers(x)  # (H, W, N_sample, D/2)
        rgb = self.fc_rgb(x)

        return rgb

    def render_img(self, coors, raw_shapes, alpha_a, alpha_l):
        H, W, Z = raw_shapes
        H, W, Z = int(alpha_a * H), int(alpha_a * W), int(alpha_l * Z)
        extend_shape = (coors.dim() - 1)*[1] + [3, 3, 3] + [1]
        tem_coors = coors.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).repeat(extend_shape)
        delta_x, delta_y, delta_z = np.meshgrid([-2/H, 0, 2/H], [-2/W, 0, 2/W], [-2/Z, 0, 2/Z], indexing="ij")
        delta = np.stack([delta_x, delta_y, delta_z], axis=-1)
        neighboring_coors = tem_coors + torch.from_numpy(delta).to(coors.device)
        pred_rgb = self.forward(neighboring_coors.float()).squeeze()
        pred_rgb = self.psf_conv(pred_rgb, 2/H, 2/W, 2/Z)
        pred_rgb = torch.sum(pred_rgb, dim=[-3, -2, -1])
        if self.is_train:
            pred_rgb = self.attenuate(coors[..., -1], pred_rgb)  # TODO: This should be commented when testing
        return pred_rgb

    def render_test_img(self, coors, *paras):
        pred_rgb = self.forward(coors).squeeze()
        # pred_rgb = self.psf_conv(pred_rgb)
        # pred_rgb = torch.sum(pred_rgb, dim=[-3, -2, -1])

        return pred_rgb