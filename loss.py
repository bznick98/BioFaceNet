import torch

def loss(predicts, targets):
    """
    total loss for BioFaceNet, which consists of 4 losses
    @input:
        predicts: a dict
            - appearance: predicted appearance map
            - b: predicted 2-dim camera sensitivity parameters
            - spec: predicted specular map
            - shading: predicted shading
        targets: a dict
            - appearance: target appearance map
            - shading: target shading
            - mask: face region mask
    @output:
        total_loss
    """
    # weights for app, prior, spars, shad loss
    w = [1e-3, 1e-4, 1e-5, 1e-5]

    total_loss =\
        appearance_loss(predicts['appearance'], targets['appearance'], targets['mask']) * w[0] +\
        prior_loss(predicts['b']) * w[1] +\
        sparsity_loss(predicts['spec']) * w[2] +\
        shading_loss(predicts['shading'], targets['shading'], targets['mask']) * w[3]

    return total_loss

def appearance_loss(appearance, target_appearance, mask):
    """
    appearance reconstruction L2-loss
    loss = ||i_{linRGB} - i_{linObs}||^2
    """
    diff = (appearance - target_appearance) * mask
    return torch.sum(diff**2)
    

def prior_loss(b):
    """
    camera prior L2-loss on sensitivity parameters
    loss = ||b||^2
    """
    return torch.sum(b**2)

def sparsity_loss(spec):
    """
    specularity L1-loss
    loss = ||spec||^1
    """
    return torch.sum(spec)

def shading_loss(shading, target_shading, mask):
    """
    shading L2-loss
    """
    diff = (shading - target_shading) * mask
    return torch.sum(diff**2)