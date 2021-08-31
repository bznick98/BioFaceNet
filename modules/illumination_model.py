"""
illumincation (physical lighting) model described in the paper
"""

import numpy as np
import torch
import scipy.io

def read_illum(
    Apath="utils/illumA.mat",
    Dpath="utils/illumDmeasured.mat",
    Fpath="utils/illF.mat"
):
    """
    read illuminant A, D, F
    @input:
        Apath, Dpath, Fpath: filepaths for each illuminant matrix
    @output:
        illumA:
        illumDNorm:
        illumFNorm:
    """
    illumA = torch.Tensor(list(scipy.io.loadmat(Apath)['illumA']))
    illumDmeasured = torch.Tensor(list(scipy.io.loadmat(Dpath)['illumDmeasured']))
    illumF = torch.Tensor(list(scipy.io.loadmat(Fpath)['illF']))

    illumA = illumA / torch.sum(illumA)
    
    illumDNorm = torch.zeros(illumDmeasured.shape)
    for i in range(22):
        illumDNorm[..., i] = illumDmeasured[..., i] / torch.sum(illumDmeasured[..., i])

    illumFNorm = torch.zeros(illumF.shape)
    for i in range(12):
        illumFNorm[..., i] = illumF[..., i] / torch.sum(illumF[..., i])

    # A: (1, 1, 33), D: (22, 33), F: (1, 33, 12)
    # print(illumA.shape, illumDmeasured.shape, illumF.shape)
    # print(illumA.shape, illumDNorm.shape, illumFNorm.shape)
    return illumA, illumDNorm, illumFNorm



def illumination_model(weightA, weightD, Fweights, CCT, illumA, illumDNorm, illumFNorm):
    """
    create illumination model from CIE standard illuminants: A, D, F
    @input:
        weightA: Nx1, 1-dim weight of illuminant A
        weightD: Nx1, 1-dim weight of illuminant D
        Fweights: Nx12, 12-dim weights, wF1~wF12
        CCT: Nx1, Camera Color Temperature i guess? the t term,
            probably discretized into 22 temperatures spaced by 10K
            from 4K to 25K. Estimation might be float, round to int
        illumA: 1x1x33, vector 
        illumDNorm: 22x33, vector
        illumFNorm: 1x33x12, vector
    @output:
        e: Nx33x1x1, 33-dim vector for each discretized wavelength
    """
    weighted_A = weightA * torch.squeeze(illumA)
    weighted_D = weightD * torch.squeeze(illumDNorm[torch.round(CCT).detach().long()-1], dim=1)
    weighted_F = torch.sum(torch.matmul(torch.squeeze(illumFNorm, dim=0), torch.diag_embed(Fweights)), dim=-1)
    
    e = weighted_A + weighted_D + weighted_F

    # Nx33x1x1
    return e[..., None, None]


# ONLY FOR TESTING:
if __name__ == '__main__':
    read_illum()
