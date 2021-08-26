"""
train BioFaceNet using part of CelebA dataset
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim import SGD

from datasets.celebA import CelebADataLoader
from BioFaceNet import BioFaceNet
from loss import loss

# argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--show', action='store_true', help='if enabled, plot 5 sample images')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

def train(args):
    # auto enable gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataloader
    dataloader = CelebADataLoader().loader

    # network
    model = BioFaceNet()

    # loss


    # optimizer
    optim = SGD(
        model.parameters(),
        lr=0.001
    )


    # training





if __name__ == '__main__':
    train(args)

    if args.show:
        CelebADataLoader().show_sample()

    # # temp visualize an input image
    # dataloader = CelebADataLoader().loader
    # for i, (image, _) in enumerate(dataloader):
    #     print(i)
    #     print(image.size())
    #     plt.imshow(image[0].permute(1,2,0))
    #     plt.show()
    #     break