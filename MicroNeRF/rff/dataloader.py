import torch

from torch import Tensor
from torch.utils.data.dataset import TensorDataset


def rectangular_coordinates(size: tuple) -> Tensor:
    r"""Creates a tensor of equally spaced coordinates for use with an image or volume

    Args:
        size (tuple): shape of the image or volume

    Returns:
        Tensor: tensor of shape :math:`(*\text{size}, \text{len(size)})`
    """
    def linspace_func(nx): return torch.linspace(0.0, 1.0, nx)
    linspaces = map(linspace_func, size)
    coordinates = torch.meshgrid(*linspaces, indexing='ij')
    return torch.stack(coordinates, dim=-1)
