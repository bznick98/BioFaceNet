"""
Make inference using trained model
"""
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import re

from PIL import Image
from BioFaceNet import BioFaceNet

# argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="checkpoints/", help='image filepath for model input')
parser.add_argument('--epoch', type=int, default=-1, help='specifies which epoch to use under checkpoints/')
parser.add_argument('--image_path', type=str, default="utils/test_img.png", help='image filepath for model input')
parser.add_argument('--output_path', type=str, default="predicted_output/", help='directory for training datasets')
args = parser.parse_args()


def predict(args):
    # auto enable gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load input image as PIL Image
    image = Image.open(args.image_path).convert('RGB').resize((64, 64))
    # convert to [0...1] as floats
    image = torchvision.transforms.functional.to_tensor(image)[None, ...]

    print(image.shape)

    # init model
    model = BioFaceNet(device=device)
    model.eval()

    # get which saved model to use, default use last epoch
    model_list = os.listdir(args.model_dir)
    epoch_list = [int(re.findall(r'[0-9]+', name)[0]) for name in model_list] # *****_20.pt, get epoch numbers as list
    last_epoch = sorted(epoch_list)[-1]
    print(last_epoch)
    if args.epoch == -1:
        epoch_to_use = last_epoch
    else:
        epoch_to_use = args.epoch
        assert args.epoch <= last_epoch

    model_base_name = "model_checkpoint_" + str(epoch_to_use) + ".pt"
    model_path = os.path.join(args.model_dir, model_base_name)
    print("Using trained model: {}".format(model_path))

    # load weights (though this loaded more things)
    model.load_state_dict(torch.load(model_path)['state_dict'])

    # infer
    fmel, fblood, shading, specular, b, lighting_params = model(image)

    # decode
    appearance, shading, specular, _ = model.decode(fmel, fblood, shading, specular, b, lighting_params)

    model.visualize_output(image, appearance, shading, specular, fmel, fblood)



if __name__ == '__main__':
    predict(args)