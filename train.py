"""
train BioFaceNet using part of CelebA dataset
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

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
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--test_forward', action='store_true', help='if enabled, test forward pass by feeding 1 image')
parser.add_argument('--viz', action='store_true', help='if enabled, images of target/model output will be plotted every batch')
parser.add_argument('--save_dir', type=str, default="checkpoints/", help='directory for saving trained model')
parser.add_argument('--data_dir', type=str, default="data/", help='directory for training datasets')
args = parser.parse_args()


def train(args):
    # make directory for checkpoints saving
    os.makedirs(args.save_dir, exist_ok=True)

    # auto enable gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data path lists
    inmc_list = [
        'zx_7_d10_inmc_celebA_20.hdf5',
        # 'zx_7_d10_inmc_celebA_05.hdf5',
    ]
    lrgb_list = [
        'zx_7_d3_lrgb_celebA_20.hdf5',
        # 'zx_7_d3_lrgb_celebA_05.hdf5',
    ]

    # inserting data dir in the front of filename
    inmc_list = [os.path.join(args.data_dir, fn) for fn in inmc_list]
    lrgb_list = [os.path.join(args.data_dir, fn) for fn in lrgb_list]

    # trainin dataloader
    train_loader = CelebADataLoader(inmc_list, lrgb_list).loader

    # network
    model = BioFaceNet(device=device)    

    # optimizer
    optim = SGD(
        model.parameters(),
        lr=args.lr
    )

    # training
    for epoch in range(args.epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                # get batch items
                image, normal, mask, actual_shading, spherical_harmonics_param = batch

                # forward pass
                fmel, fblood, pred_shading, pred_specular, b, lighting_params = model(image)

                # decode (physcial model based)
                appearance, pred_shading, pred_specular, b = model.decode(fmel, fblood, pred_shading, pred_specular, b, lighting_params)

                # visualize training progress
                if args.viz:
                    model.visualize_training_progress(image, actual_shading, mask, appearance, pred_shading, pred_specular, fmel, fblood)

                # pack predicted items for loss computation
                predicts = {
                    'appearance': appearance,
                    'b': b,
                    'specular': pred_specular,
                    'shading': pred_shading
                }

                targets = {
                    'appearance': image,
                    'shading': actual_shading,
                    'mask': mask
                }

                # compute loss
                batch_loss = loss(predicts, targets)

                # reset optimizer & backprop
                optim.zero_grad()
                batch_loss.backward()
                optim.step()

                # update info
                tepoch.set_postfix(epoch="{}/{}".format(epoch+1, args.epochs), loss=batch_loss.cpu().detach().numpy())

        # save model each epoch
        ckpt_filename = "model_checkpoint_{}.pt".format(epoch)
        save_path = os.path.join(args.save_dir, ckpt_filename)
        state = {
            'epoch':epoch,
            'state_dict':model.state_dict(), # use model.load_state_dict(torch.load(XX)) when resume training
        }
        torch.save(state, save_path)



if __name__ == '__main__':
    train(args)

    if args.show:
        CelebADataLoader().show_sample()

    # Test forward pass
    if args.test_forward:
        # data path lists
        inmc_list = [
            'data/zx_7_d10_inmc_celebA_20.hdf5',
            # 'data/zx_7_d10_inmc_celebA_05.hdf5',
        ]
        lrgb_list = [
            'data/zx_7_d3_lrgb_celebA_20.hdf5',
            # 'data/zx_7_d3_lrgb_celebA_05.hdf5',
        ]

        # trainin dataloader
        train_loader = CelebADataLoader(inmc_list, lrgb_list).loader

        # network
        model = BioFaceNet()

        for batch in train_loader:
            image, normal, mask, actual_shading, spherical_harmonics_param = batch
            image, normal, mask, actual_shading, spherical_harmonics_param = image[0][None,...], normal[0][None,...], mask[0][None,...], actual_shading[0][None,...], spherical_harmonics_param[0][None,...]
            output = model(image)

            fmel, fblood, shading, specular, b, lighting_params = output

            print(fmel.shape, fblood.shape, shading.shape, specular.shape, b.shape, lighting_params.shape)
            # plt.imshow(fmel[0][0].detach().numpy())
            # plt.show()
            print(lighting_params)
            break
