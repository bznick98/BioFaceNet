"""
illumincation (physical lighting) model described in the paper
"""

import numpy as np


def illumination_model(weightA, weightD, Fweights, CCT, illumA, illumDNorm, illumFNorm):
    """
    create illumination model from CIE standard illuminants: A, D, F
    @input:
        weightA: 1-dim weight of illuminant A
        weightD: 1-dim weight of illuminant D
        Fweights: 12-dim weights, wF1~wF12
        CCT: Camera Color Temperature i guess? the t term,
            probably discretized into 22 temperatures spaced by 10K
            from 4K to 25K. Estimation might be float, round to int
        illumA: 33-dim vector 
        illumDNorm: 33x22-dim vector
        illumFNorm: 33x12-dim vector
    @output:
        e: 33-dim vector for each discretized wavelength
    """
    weighted_A = weightA * illumA
    weighted_D = weightD * illumDNorm[CCT-1]
    weighted_F = np.sum(np.matmul(illumFNorm, np.diag(Fweights)), axis=1)
    
    return weighted_A + weighted_D + weighted_F