import os
import numpy as np

import nibabel as nib
import torch
import torch.nn.functional as F

def load_nii_data(file_name):
    nii_data = nib.load(file_name)
    data = nii_data.get_fdata()
    header = nii_data.header

    return data, header

def sample_points_by_intensity(X, N):
    """
    Sample N points from a 3D grayscale array based on intensity values.

    Parameters:
    X : 3D numpy array
        The input 3D grayscale array
    N : int
        Number of points to sample

    Returns:
    coords : Nx3 numpy array
        Coordinates of sampled points
    """
    weights = X.ravel()
    weights = torch.sqrt(weights + weights.mean()).double()
    total_elements = weights.shape[0]
    sampled_indices = np.random.choice(total_elements, size=N)  #, p=weights / weights.sum(), replace=False)

    coords = np.column_stack(np.unravel_index(sampled_indices, X.shape))
    intensities = X[coords[..., 0], coords[..., 1], coords[..., 2]]

    return coords, intensities

class Micro3D:
    def __init__(self, img_file, interval, alpha_a, alpha_l):
        self.img_file = img_file
        self.alpha_a = alpha_a
        self.alpha_l = alpha_l
        self.interval = interval
        self.img = self._load_img(img_file)
        self.sz = self.img.shape[-1]
        self.interval = None

    def _load_img(self, img_file):
        if img_file.endswith(".gz"):
            nii_img = nib.load(img_file)
            nii_header = nii_img.header
            img = nii_img.get_fdata().astype(np.uint8)
            self.header = nii_header
            self.x_res = nii_header["pixdim"][1]
            self.z_res = nii_header["pixdim"][3]
        else:
            img = np.load(img_file)
        # img = self.normalize_3d_array(img) # / 255.0
        low, high = np.percentile(img, [0.1, 99.95])
        img = np.clip(img, low, high)
        img = ((img - img.min()) / (img.max() - img.min()))  # * 255).astype(np.uint8)
        return torch.from_numpy(img).float()

    def normalize_3d_array(self, img, low_percentile=1, high_percentile=99.9, target_min=0, target_max=255):
        low, high = np.percentile(img, [low_percentile, high_percentile])
        normalized = np.clip(img, low, high)
        normalized = (normalized - low) / (high - low)
        return (normalized * (target_max - target_min) + target_min).astype(np.uint8)

    def get_train_sample(self, xy_points=100):
        # input_slices = list(range(self.sz))
        # input_img = self.img[..., input_slices]
        input_coors, input_values = sample_points_by_intensity(self.img, xy_points * xy_points)
        input_coors[..., :2] = input_coors[..., :2] * self.alpha_a
        input_coors[..., -1] = input_coors[..., -1] * self.alpha_l
        input_coors = self._normalize_pos(input_coors.astype(np.float32))
        input_coors = torch.from_numpy(input_coors).float()

        return input_values, input_coors

    def get_eval_sample(self, xy_points=100):
        # input_slices = list(range(0, self.sz, self.interval))
        # pred_slices = list(set(range(self.sz)) - set(input_slices))
        #
        # pred_img = self.img[..., pred_slices]
        pred_coors, pred_values = sample_points_by_intensity(self.img, xy_points * xy_points)
        pred_coors[..., :2] = pred_coors[..., :2] * self.alpha_a
        pred_coors[..., -1] = pred_coors[..., -1] * self.alpha_l
        pred_coors = self._normalize_pos(pred_coors.astype(np.float32))
        pred_coors = torch.from_numpy(pred_coors).float()

        return pred_values, pred_coors

    def _normalize_pos(self, coors):
        H, W, Z = self.img.shape
        coors[..., 0] = 2 * coors[..., 0] / int(H * self.alpha_a) - 1
        coors[..., 1] = 2 * coors[..., 1] / int(W * self.alpha_a) - 1
        coors[..., 2] = 2 * coors[..., 2] / int(Z * self.alpha_l) - 1
        return coors

    def _get_volume_pos(self):
        H, W, Z = self.img.shape
        target_H, target_W, target_Z = int(H * self.alpha_a), int(W * self.alpha_a), int(Z * self.alpha_l)
        coors = self._get_pos(target_H, target_W, target_Z)
        return coors

    def _get_pos(self, H, W, Z):
        x_coor = list(np.linspace(-1, 1, H))
        y_coor = list(np.linspace(-1, 1, W))
        z_coor = list(np.linspace(-1, 1, Z))
        x_coors, y_coors, z_coors = np.meshgrid(x_coor, y_coor, z_coor, indexing="ij")
        coors = np.stack([x_coors, y_coors, z_coors], axis=-1)

        return torch.from_numpy(coors).float()



