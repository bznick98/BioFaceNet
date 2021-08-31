"""
Biophysical spectral skin model
"""

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io

def read_new_skin_color(filepath="utils/Newskincolour.mat"):
    """
    read from Matlab mat file
    @input:
        filepath: path to mat file
    @output:
        new_skin_color: 256x256x33, matrix used in Matlab implementation
    """
    # No idea of how Newskincolour matrix is calculated, load from file
    new_skin_color = torch.Tensor(list(scipy.io.loadmat(filepath)['Newskincolour']))

    return new_skin_color

def biophysical_model(fmel, fblood, new_skin_color):
    """
    biophyscial spectral skin model described in the paper
    @input: TODO
        fmel: Nx1xHxW, predicted melanin map
        fblood: Nx1xHxW, precdicted haemoglobin map
        new_skin_color: 256x256x33
    @output:
        R: Nx33xHxW
    """
    # stack fmel and fblood
    biophysical_map = torch.cat([fmel, fblood], dim=1)

    # repeatedly stack new_skin_color N(batchsize) times
    batchsize = fmel.shape[0]
    new_skin_color = new_skin_color[None, ...].repeat(batchsize,1,1,1)
    new_skin_color = torch.moveaxis(new_skin_color, -1, 1)

    # bilinear sample used in Matlab implementation, pytorch has grid_sample
    # move size 2 axis to the last place since grid_sample requires that
    R = F.grid_sample(new_skin_color, torch.moveaxis(biophysical_map, 1, -1), align_corners=False)

    # Nx33xHxW
    return R


# ONLY FOR TESTING
if __name__ == '__main__':
    read_new_skin_color()