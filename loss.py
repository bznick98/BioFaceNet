import torch
import numpy as np
import matplotlib.pyplot as plt

def loss(predicts, targets):
    """
    total loss for BioFaceNet, which consists of 4 losses
    @input:
        predicts: (a dict)
            - 'appearance': predicted appearance map
            - 'b': predicted 2-dim camera sensitivity parameters
            - 'specular': predicted specular map
            - 'shading': predicted shading
        targets: (a dict)
            - 'appearance': target appearance map
            - 'shading': target shading
            - 'mask': face region mask
    @output:
        total_loss
    """
    # weights for app, prior, spars, shad loss
    w = [1e-3, 1e-4, 1e-5, 1e-5]

    total_loss =\
        appearance_loss(predicts['appearance'], targets['appearance'], targets['mask']) * w[0] +\
        prior_loss(predicts['b']) * w[1] +\
        sparsity_loss(predicts['specular']) * w[2] +\
        shading_loss(predicts['shading'], targets['shading'], targets['mask']) * w[3]

    return total_loss

def appearance_loss(appearance, target_appearance, mask):
    """
    appearance reconstruction L2-loss
    loss = ||i_{linRGB} - i_{linObs}||^2
    """
    diff = (appearance - target_appearance) * mask
    # print("predicted: ", appearance)
    # print("target: ", target_appearance)

    # print("APPPPPPSHAPE: ", np.moveaxis(appearance[0].detach().numpy(), 0,-1).shape)

    # plt.imshow(np.moveaxis(appearance[0].detach().numpy(), 0,-1))
    # plt.show()

    # print("DIFFFFFF: =", diff.shape)

    loss = torch.sum(diff**2)
    # print("APPEARANCE LOSS: ", loss)
    return loss

def prior_loss(b):
    """
    camera prior L2-loss on sensitivity parameters
    loss = ||b||^2
    """
    loss = torch.sum(b**2)
    # print("PRIOR LOSS: ", loss)
    return loss

def sparsity_loss(spec):
    """
    specularity L1-loss
    loss = ||spec||^1
    """
    loss = torch.sum(spec)
    # print("SPARS LOSS: ", loss)
    return loss

def shading_loss(shading, target_shading, mask):
    """
    shading L2-loss
    TODO: there is an extra linear regression of scaling applied to predicted shading,
    not sure if Matlab version implemented it
    """
    scale = torch.sum(torch.sum(target_shading * shading * mask, dim=2, keepdim=True), dim=3, keepdim=True) / torch.sum(torch.sum(shading * shading * mask, dim=2, keepdim=True), dim=3, keepdim=True)
    # FIXME: after scale, shading value will explode, remove scaling temporarily
    shading = shading * scale

    diff = (shading - target_shading) * mask
    loss = torch.sum(diff**2)

    # print("SHADING LOSS: ", loss)
    return loss