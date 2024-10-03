import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import nibabel as nib

import torch
import torch.nn.functional as F

from inr import OfficialNerfMean
from dataset import Micro3D
from tool import perceptual_loss

def set_randomness(args):
    if args.true_rand is False:
        random.seed(args.rand_seed)
        np.random.seed(args.rand_seed)
        torch.manual_seed(args.rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default=50000, type=int)
    parser.add_argument("--img_file", type=str, required=True)
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--interval", default=1, type=int)
    parser.add_argument("--alpha_a", default=1, type=float)
    parser.add_argument("--alpha_l", default=3, type=float)
    parser.add_argument("--xy_points", default=50, type=int)
    parser.add_argument("--test_size", default=50, type=int)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--is_train", action='store_true')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--pos_in_dims", default=512, type=int)

    return parser.parse_args()

def main(paras):

    dataset = Micro3D(img_file=paras.img_file, interval=paras.interval, alpha_a=paras.alpha_a, alpha_l=paras.alpha_l)
    model = OfficialNerfMean(pos_in_dims=paras.pos_in_dims, dir_in_dims=0, D=64)
    model = model.cuda()

    optimizer_nerf = torch.optim.Adam(model.parameters(), lr=paras.lr)
    scheduler_nerf = torch.optim.lr_scheduler.MultiStepLR(optimizer_nerf,
                                                          milestones=[5000, 20000],  # TODO: refine learning strategy
                                                          gamma=0.1)

    for i_step in range(1, paras.step+1, 1):
        model.train()
        input_slices, input_pos = dataset.get_train_sample(xy_points=paras.xy_points)
        input_slices, input_pos = input_slices.cuda(), input_pos.cuda()
        pred_rgb = model.render_img(input_pos, dataset.img.shape, dataset.alpha_a, dataset.alpha_l).squeeze()
        loss = F.mse_loss(input_slices, pred_rgb)
        loss.backward()
        optimizer_nerf.step()
        optimizer_nerf.zero_grad()
        scheduler_nerf.step()
        if i_step % 10 == 0:
            print("Step: {:>5d}/{:>5d}, MSE loss: {:>10.5f}".format(i_step, paras.step, loss.item()))

        best_loss = 1000
        if i_step % 50 == 0:
            with torch.no_grad():
                model.eval()
                implicit_slices, implicit_pos = dataset.get_eval_sample(xy_points=paras.xy_points)
                implicit_slices, implicit_pos = implicit_slices.cuda(), implicit_pos.cuda()
                pred_rgb = model.render_img(implicit_pos, dataset.img.shape, dataset.alpha_a, dataset.alpha_l).squeeze()
                loss = F.mse_loss(implicit_slices, pred_rgb)
                print("Evaluation Step: {:>5d}/{:>5d}, MSE loss: {:>10.5f}".format(i_step, paras.step, loss.item()))
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    model_file = os.path.join(paras.save_dir,
                                              os.path.basename(paras.img_file).split(".")[0] + "_stepBest.pth")
                    torch.save(model.state_dict(), model_file)


        if i_step % 10000 == 0:
            model_file = os.path.join(paras.save_dir,
                                      os.path.basename(paras.img_file).split(".")[0] + "_step{}.pth".format(i_step))
            torch.save(model.state_dict(), model_file)
            print("Model saved to {}".format(model_file))

    model_file = os.path.join(paras.save_dir, os.path.basename(paras.img_file).split(".")[0] + "_step{}.pth".format(i_step))
    torch.save(model.state_dict(), model_file)
    print("Model saved to {}".format(model_file))

def test(paras):

    model = OfficialNerfMean(pos_in_dims=paras.pos_in_dims, dir_in_dims=0, D=64, is_train=paras.is_train)
    model.cuda()
    model.load_state_dict(torch.load(paras.model_file))
    model.eval()

    dataset = Micro3D(paras.img_file, interval=paras.interval, alpha_a=paras.alpha_a, alpha_l=paras.alpha_l)
    output = []
    size = paras.test_size
    with torch.no_grad():
        img_H, img_W, img_Z = dataset.img.shape
        H, W, Z = int(img_H * paras.alpha_a), int(img_W * paras.alpha_a), int(img_Z * paras.alpha_l)
        output = np.zeros([H, W, Z])
        coors = dataset._get_volume_pos()
        for start_h in tqdm(range(0, int(H // size + 1) * size, size), desc="Rendering volume..."):
            for start_w in range(0, int(W // size + 1) * size, size):
                for start_z in range(0, int(Z // size + 1) * size, size):
                    pos = coors[start_h:start_h+size, start_w:start_w+size, start_z:start_z+size].cuda()
                    # TODO: Compare render_img and render_test_img for efficiency of deconvolution.
                    pred_rgb = model.render_img(pos, dataset.img.shape, dataset.alpha_a, dataset.alpha_l).squeeze().cpu().numpy()
                    pred_rgb = np.clip(pred_rgb, 0, 255)  # * 255
                    output[start_h:start_h+size, start_w:start_w+size, start_z:start_z+size] = pred_rgb
    output = ((output - output.min()) / (output.max() - output.min()) * 255).astype(np.uint8)
    n_min, n_max = np.percentile(output, [0, 99.9])
    output = np.clip(output, 0, n_max)
    output = (output / n_max * 255).astype(np.uint8)
    output_file = os.path.join(paras.save_dir, os.path.basename(paras.img_file).split(".")[0] + "_reconstruct.nii.gz")
    output_nii = nib.Nifti1Image(output, np.eye(4))
    output_nii.header.set_xyzt_units(xyz=3, t=8)
    output_nii.header["pixdim"] = [1.0, dataset.x_res/dataset.alpha_a, dataset.x_res/dataset.alpha_a, dataset.z_res/dataset.alpha_l, 0., 0., 0., 0.]
    # output_nii.header["pixdim"] = [1.0, 0.09, 0.09, 0.21/paras.alpha_l, 0., 0., 0., 0.]
    nib.save(output_nii, output_file)
    print("Reconstructed data saved to {}".format(output_file))

if __name__ == "__main__":
    paras = parse_args()
    # check ratio
    ratio = os.path.basename(paras.img_file).split(".")[0].split("_")[-1]
    ratio = float(ratio)
    paras.alpha_l = ratio
    if paras.is_train:
        main(paras)
    else:
        test(paras)