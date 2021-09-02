import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from torchvision import transforms

from modules.camera_model import camera_model, camera_PCA, read_camspec
from modules.illumination_model import illumination_model, read_illum
from modules.biophysical_model import biophysical_model, read_new_skin_color


class BioFaceNet(nn.Module):
    def __init__(self, device=torch.device('cpu'), viz_big=False):
        """
        basically a UNet plus fully connected branch from the lowest resolution
        """
        super().__init__()

        # cpu/gpu
        self.device = device

        # if set to True, visualization will plot larger image, recommend toggle this if using Google Colab
        self.viz_big = viz_big

        # downsampling layers, down path double convolution
        self.down1 = self.double_conv(3, 32)
        self.down2 = self.double_conv(32, 64)
        self.down3 = self.double_conv(64, 128)
        self.down4 = self.double_conv(128, 256)
        self.down5 = self.double_conv(256, 512)


        # maxpooling
        self.maxpool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # upsampling layers, transpose convolution (someother uses upsampling)
        self.t_conv1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2)
        self.bn_t1 = nn.BatchNorm2d(256)
        self.t_conv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)
        self.bn_t2 = nn.BatchNorm2d(128)
        self.t_conv3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2)
        self.bn_t3 = nn.BatchNorm2d(64)
        self.t_conv4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2)
        self.bn_t3 = nn.BatchNorm2d(32)

        # up path double convolution
        self.up1 = self.double_conv(512, 256)
        self.up2 = self.double_conv(256, 128)
        self.up3 = self.double_conv(128, 64)
        self.up4 = self.double_conv(64, 32)

        # output conv layer
        self.out = nn.Conv2d(
            in_channels=32,
            out_channels=4,
            kernel_size=3,
            padding='same'
        )

        # fully connected branch on the lowest resolution
        #   for predicting vector quantities (b, wA, wD, t, wF1, ..., wF12)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128), 
            nn.Linear(128, 17), # bSize(2) + LightVectorSize(15)
        )

        # modules initialization
        # camera model initialization
        rgbCMF = read_camspec()
        self.camera_PC, self.camera_mean, self.camera_eigenvalues = camera_PCA(rgbCMF)
        self.camera_PC, self.camera_mean, self.camera_eigenvalues = self.camera_PC.to(self.device), self.camera_mean.to(self.device), self.camera_eigenvalues.to(self.device)

        # biophysical model initialization
        self.new_skin_color = read_new_skin_color()
        self.new_skin_color = self.new_skin_color.to(self.device)

        # Tmatrix initialization 128x128x9
        self.Tmatrix = self.read_Tmatrix()
        self.Tmatrix = self.Tmatrix.to(self.device)

    def forward(self, image):
        """
        forward passing
        @input:
            image: a batch of rgb images
        @output:
            fmel: Nx1xHxW, melanin map
            fblood: Nx1xHxW, haemoglobin map
            shading: Nx1xHxW, diffuse? shading map
            specular: Nx1xHxW, specular shading map
            b: Nx2, 2-dim vector of camera sensitivity parameters
            lighting_params: Nx15, 15-dim vector of lighting parameters
        """
        # print("Input image size: ", image.shape)
        # encoder
        x1 = self.down1(image) #
        x2 = self.maxpool_2x2(x1)

        # print("Size after 1: ", x2.shape)

        x3 = self.down2(x2) #
        x4 = self.maxpool_2x2(x3)

        # print("Size after 2: ", x4.shape)

        x5 = self.down3(x4) #
        x6 = self.maxpool_2x2(x5)

        # print("Size after 3: ", x6.shape)

        x7 = self.down4(x6) # 
        x8 = self.maxpool_2x2(x7)

        # print("Size after 4: ", x8.shape)

        x9 = self.down5(x8)

        # lowest resolution output (1, 512, 4, 4) if input is 64x64
        low_res = x9

        # print("Size of lowest resolution: ", x9.shape)

        # decoder
        x = self.t_conv1(x9)
        # concatenate x7 (after crop) with x 
        y = self.crop_img(x7, x)
        x = torch.cat([x, y], 1)
        x = self.up1(x)

        x = self.t_conv2(x)
        # concatenate x5 (after crop) with x 
        y = self.crop_img(x5, x)
        x = torch.cat([x, y], 1)
        x = self.up2(x)

        x = self.t_conv3(x)
        # concatenate x3 (after crop) with x 
        y = self.crop_img(x3, x)
        x = torch.cat([x, y], 1)
        x = self.up3(x)

        x = self.t_conv4(x)
        # concatenate x1 (after crop) with x 
        y = self.crop_img(x1, x)
        x = torch.cat([x, y], 1)
        x = self.up4(x)
   
        # output
        x = self.out(x)
        # print(x.shape)

        # x.shape = (B, 4, W, H)
        fmel = x[:, 0, :, :][:, None, ...]
        fblood = x[:, 1, :, :][:, None, ...]
        shading = x[:, 2, :, :][:, None, ...]
        specular = x[:, 3, :, :][:, None, ...]

        # fully connected branch from the lowest resolution
        vector_output = self.fc(low_res)
        # print(vector_output.shape)
        b = vector_output[:, :2]
        lighting_params = vector_output[:, 2:]

        # print("--------------------------!Check Shape Correctness!---------------------------")
        # print(fmel.shape, fblood.shape, shading.shape, specular.shape, b.shape, lighting_params.shape)

        # 4 feature maps, 1 2-dim vector of camera param, 1 15-dim vector of lighting 
        return fmel, fblood, shading, specular, b, lighting_params

    def double_conv(self, in_ch, out_ch, padding='same'):
        """
        double conv used in UNet
        @input:
            in_ch: number of input channels(filters)
            out_ch: number of output channels
            padding: default = 'same' since input image size is relatively small
        @output:
            double_conv_layer: a nn sequential layer consists
                of double convolutional layers
        """
        double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return double_conv

    def crop_img(self, tensor, target_tensor):
        """
        used for cropping feature maps when concatenating down/up layers
        """
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]

        delta = (tensor_size - target_size) // 2
        # center crop
        return tensor[:, :, delta:tensor_size-delta, delta: tensor_size-delta]

    def decode(self, fmel, fblood, shading, specular, b, lighting_params):
        """
        decoding sRGB faces from physical-based parameters,
        as well as scaling predicted maps for loss computation
        @input: TODO
            fmel: Nx1xHxW
            fblood: Nx1xHxW
            shading: Nx1xHxW
            specular: Nx1xHxW
            b: Nx2
            lighting_params: Nx15
        @output:
            appearance: Nx3xHxW, reconstructed face sRGB image from parameters
            shading: Nx1xHxW, shading after scaling
            specular: Nx1xHxW, specularities after scaling
            b: Nx2, for loss computation
        """
        # print("[[[BEFORE DECODE, fmel MAX/MIN]]]: ", torch.max(fmel), torch.min(fmel))
        # print("[[[BEFORE DECODE, fblood MAX/MIN]]]: ", torch.max(fblood), torch.min(fblood))
        # print("[[[BEFORE DECODE, shading MAX/MIN]]]: ", torch.max(shading), torch.min(shading))
        # print("[[[BEFORE DECODE, spec MAX/MIN]]]: ", torch.max(specular), torch.min(specular))
        # print("[[[BEFORE DECODE, b MAX/MIN]]]: ", torch.max(b), torch.min(b))
        # print("[[[BEFORE DECODE, light_param MAX/MIN]]]: ", torch.max(lighting_params), torch.min(lighting_params))

        # print("<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>")
        # scale output
        weightA, weightD, CCT, Fweights, b, fmel, fblood, shading, specular, BGrid = self.scale_output(lighting_params, b, fmel, fblood, shading, specular)

        # print("[[[AFTER DECODE, fmel MAX/MIN]]]: ", torch.max(fmel), torch.min(fmel))
        # print("[[[AFTER DECODE, fblood MAX/MIN]]]: ", torch.max(fblood), torch.min(fblood))
        # print("[[[AFTER DECODE, shading MAX/MIN]]]: ", torch.max(shading), torch.min(shading))
        # print("[[[AFTER DECODE, spec MAX/MIN]]]: ", torch.max(specular), torch.min(specular))
        # print("[[[AFTER DECODE, b MAX/MIN]]]: ", torch.max(b), torch.min(b))
        # print("[[[AFTER DECODE, light_param MAX/MIN]]]: ", torch.max(lighting_params), torch.min(lighting_params))
        # print("[[[AFTER DECODE, CCT MAX/MIN]]]: ", torch.max(CCT), torch.min(CCT))
        # print("[[[AFTER DECODE, weightA MAX/MIN]]]: ", torch.max(weightA), torch.min(weightA))
        # print("[[[AFTER DECODE, weightD MAX/MIN]]]: ", torch.max(weightD), torch.min(weightD))


        # print("-----------------!Scale Output Shape Correctness!-----------------")
        # print(weightA.shape, weightD.shape, CCT.shape, Fweights.shape, b.shape, fmel.shape, fblood.shape, shading.shape, specular.shape, BGrid.shape)

        # illuminant model
        illumA, illumDNorm, illumFNorm = read_illum()
        # move to specified device
        illumA, illumDNorm, illumFNorm = illumA.to(self.device), illumDNorm.to(self.device), illumFNorm.to(self.device)
        e = illumination_model(weightA, weightD, Fweights, CCT, illumA, illumDNorm, illumFNorm)

        # camera model
        Sr, Sg, Sb = camera_model(self.camera_mean, self.camera_PC, b)

        # light color
        light_color = self.light_color(e, Sr, Sg, Sb)

        # specularities
        specular = self.compute_specularities(specular, light_color)

        # Biophysical to spectral reflectance
        R = biophysical_model(fmel, fblood, self.new_skin_color)

        # image formation
        raw_img, diffuse_albedo = self.image_formation(R, Sr, Sg, Sb, e, specular, shading)

        # white balance
        wb_img = self.white_balance(raw_img, light_color)

        # raw to sRGB
        T_raw2xyz = self.findT(self.Tmatrix, BGrid)
        appearance = self.raw_to_sRGB(wb_img, T_raw2xyz) # sRGB
        
        return appearance, shading, specular, b

    def raw_to_sRGB(self, wb_img, T_raw2xyz):
        """
        color transformation pipeline: convert raw image to sRGB image (by batch)
        @input:
            wb_img: Nx3xHxW, white balanced raw image
            T_raw2xyz: Nx9x1x1, raw to xyz transformation matrix loaded from matlab's repo
        @output:
            sRGB: sRGB image (non-linear)
        """
        Ix = T_raw2xyz[:, 0, :, :] * wb_img[:, 0, :, :] + T_raw2xyz[:, 3, :, :] * wb_img[:, 1, :, :] + T_raw2xyz[:, 6, :, :] * wb_img[:, 2, :, :]
        Iy = T_raw2xyz[:, 1, :, :] * wb_img[:, 0, :, :] + T_raw2xyz[:, 4, :, :] * wb_img[:, 1, :, :] + T_raw2xyz[:, 7, :, :] * wb_img[:, 2, :, :]
        Iz = T_raw2xyz[:, 2, :, :] * wb_img[:, 0, :, :] + T_raw2xyz[:, 5, :, :] * wb_img[:, 1, :, :] + T_raw2xyz[:, 8, :, :] * wb_img[:, 2, :, :]

        # stack on new axis channel
        Ixyz = torch.cat([Ix[:, None, ...], Iy[:, None, ...], Iz[:, None, ...]], dim=1)


        T_xyz2rgb = torch.Tensor([
            [3.2406, -1.537, -0.498],
            [-0.968, 1.8758, 0.0415],
            [0.0557, -0.204, 1.0570]
        ]).to(self.device)

        # perform temp move axis for matrix multiply, Ixyz: NxHxWx3
        Ixyz = torch.moveaxis(Ixyz, 1, -1)

        # NxHxWx3 * 3x3 => NxHxWx3
        rgb_img = torch.matmul(Ixyz, T_xyz2rgb)

        # Nx3xHxW
        rgb_img = torch.moveaxis(rgb_img, -1, 1)

        # add relu before apply gamma (otherwise negative number will be raised power to 1/gamma, which -> NaN)
        rgb_img = F.relu(rgb_img)

        # apply non-linear gamma correction
        a, gamma = 0.055, 2.4
        srgb = (1 + a) * torch.pow(rgb_img, 1/gamma) - a
        return srgb
    
    def light_color(self, e, Sr, Sg, Sb):
        """
        compute color of light
        @input: TODO
            e: Nx33x1x1
            Sr, Sg, Sb: Nx33x1x1 each
        @output:
            light_color: Nx3x1x1
        """
        light_color = [torch.sum(Sr * e, dim=1)[..., None], torch.sum(Sg * e, dim=1)[..., None], torch.sum(Sb * e, dim=1)[..., None]]
        return torch.cat(light_color, dim=1)

    def white_balance(self, raw, light_color):
        """
        white balancing raw appearance
        @input:
            raw: Nx3xHxW, raw image decoded from model
            light_color: Nx3x1x1, defined in the paper
        @output:
            wb_raw: Nx3xHxW, white balanced raw appearance
        """
        r = raw[:, 0, :, :] / light_color[:, 0, :, :]
        g = raw[:, 1, :, :] / light_color[:, 1, :, :]
        b = raw[:, 2, :, :] / light_color[:, 2, :, :]

        # in order to remain 4-D
        r = r[:, None, ...]
        g = g[:, None, ...]
        b = b[:, None, ...]

        return torch.cat((r, g, b), dim=1)  # cat along channel
    
    def compute_specularities(self, specular, lightcolor):
        """
        compute specularities from specmask
        @input: TODO
            specular: Nx1xHxW, specularity maps after scaling
            lightcolor: Nx3x1x1
        @output:
            specular: Nx1xHxW
        """
        # results in shape Nx3xHxW
        specular = specular * lightcolor

        # however, above element wise mult results in Nx3xHxW
        #   instead of Nx1xHxW indicated in matlab's implementation,
        #   thus convert to grayscale to keep 1 channel (not sure if right)
        grayscale = transforms.Grayscale()
        specular = grayscale(specular)
        return specular

    def image_formation(self, R, Sr, Sg, Sb, e, specular, shading):
        """
        produce raw image from model-based parameters
        @input: TODO
            R: Nx33xHxW, biophysical model output
            Sr, Sg, Sb: Nx33x1x1 each, statistical camera model
            e: Nx33x1x1, lighting (illuminant) model
            specular: Nx1xHxW, predicted specular mask (scaled)
            shading: Nx1xHxW, predicted shading (scaled)
        @output:
            raw_img: Nx1xHxW, need to transform to sRGB before compute loss
            diffuse_albedo: Nx3xHxW
        """
        # print("=======!!!!!\n", R.shape, Sr.shape, Sg.shape, Sb.shape, e.shape, specular.shape, shading.shape)
        spectra_ref = R * e

        # sum over channel values (NxHxWxC)
        r_channel = torch.sum(spectra_ref * Sr, dim=1, keepdim=True)
        g_channel = torch.sum(spectra_ref * Sg, dim=1, keepdim=True)
        b_channel = torch.sum(spectra_ref * Sb, dim=1, keepdim=True)

        diffuse_albedo = torch.cat([r_channel, g_channel, b_channel], dim=1)

        shaded_diffuse = diffuse_albedo * shading

        raw_img = shaded_diffuse + specular

        return raw_img, diffuse_albedo


    def read_Tmatrix(self, filepath="utils/Tmatrix.mat"):
        """
        load Tmatrix used for findT
        @input:
            filepath: filepath to Tmatrix.mat (from Matlab repo)
        @output:
            Tmatrix: 128x128x9
        """
        Tmatrix = torch.Tensor(list(scipy.io.loadmat("utils/Tmatrix.mat")['Tmatrix']))

        return Tmatrix


    def findT(self, Tmatrix, BGrid):
        """
        find matrix used for RAW2XYZ
        @input:
            Tmatrix: 128x128x9
            BGrid: Nx2x1x1, camera sensitivity parameters with extra dimensions added
        @output:
            T_raw2xyz: Nx9x1x1
        """
        # duplicate N(batchsize) times of Tmatrix for the use of grid_sample
        batchsize = BGrid.shape[0]
        Tmatrix = Tmatrix[None, ...].repeat(batchsize,1,1,1)
        # move axis of size 2 of BGrid to last as grid_sample required
        BGrid = torch.moveaxis(BGrid, 1, -1)
        # move axis of Tmatrix from channel_last to channel first
        Tmatrix = torch.moveaxis(Tmatrix, -1, 1)
        T_raw2xyz = F.grid_sample(Tmatrix, BGrid, align_corners=False)
        return T_raw2xyz

    def scale_output(self, lighting_params, b, fmel, fblood, shading, spec):
        """
        perform scaling on network's output as described in the paper,
        similar to scalingNet() in Matlab implementation
        @input: 
            lighting_params: 15-dimensional vector for lighting parameters
            b: 2-dimensional vector for camera parameters
            fmel: predicted melanin map
            fblood: predicted haemoglobin map
            shading: predicted shading
            mask: face mask
            (all inputs expects a first extra dimension of batchsize)
        @output:
            weightA: used for illum. model
            weightD: used for illum. model
            Fweights: used for illum. model
            CCT: used for illum. model, color temperature, bounded [1,22]
            b: Nx2, camera sensitivity params, bounded [-3,3]
            BGrid: Nx2x1x1, b with two extra axes, scaled between [-1,1]
            fmel: scaled melanin map, bounded [-1,1]
            fblood: scaled haemoglobin map, bounded [-1,1]
            shading: scaled shading, exponentiated
            spec: scaled specularity, exponentiated
            BGrid: Nx2x1x1, b with new axes added
        """
        # Normalize lighting parameters (wA, wD, wF1~wF12) so that sum to 1
        lighting_params = F.softmax(lighting_params, dim=1)
        weightA = lighting_params[:, 0][..., None]
        weightD = lighting_params[:, 1][..., None]
        CCT = lighting_params[:, 2][..., None]
        CCT = ((22 - 1) / (1 + torch.exp(-CCT))) + 1
        # print("CCT AFTER SCALE::::: ", CCT)
        Fweights = lighting_params[:, 3:]
        b = 6 * torch.sigmoid(b) - 3            # [-3, 3]
        BGrid = b[..., None, None] / 3      # [-1 ,1]
        fmel = torch.sigmoid(fmel) * 2 - 1      # [-1, 1]
        fblood = torch.sigmoid(fblood) * 2 - 1  # [-1, 1]
        shading = torch.exp(shading)        # must positive
        spec = torch.exp(spec)              # must positive

        return weightA, weightD, CCT, Fweights, b, fmel, fblood, shading, spec, BGrid

    def visualize_training_progress(self, image, actual_shading, mask, appearance, pred_shading, pred_specular, fmel, fblood, num=5, cmap='cividis'):
        """
        visualize targets and predicted map every batch
        @input:
            same as model's outputs and targets
            num: number of samples to show (rows in plots, default=5)
            cmap: only for fmel & fblood, string of color map accepted by matplotlib (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
        @output:
            None, plotting 5x8 plots, plot will last 7 seconds and close itself
        """
        # determine visualization size, recommend (8, 6) for local, (16, 12) for Colab
        fig_size = (8, 6)
        fontsize = 7
        if self.viz_big:
            fig_size = (16, 12) # really big
            fontsize = 9

        fig, axes = plt.subplots(num, 8, figsize=fig_size)
        # fig.suptitle("Sample {} images from decode output as training progress with order:\n\
        #     Target(first three), Predicted(the rest)\n\
        #         Input Image, Computed Shading, Mask | Pred Appearance, Pred Shading, Pred Spec, Pred Fmel, Pred Fblood".format(num))
        title_list = ['Input', 'Actual Shading', 'Mask', 'Pred Face', 'Pred Shading', 'Pred Specular', 'fmel', 'fblood']
        for i in range(num):
            # add column header
            if i == 0:
                for col in range(8):
                    axes[i, col].set_title(title_list[col], fontsize=fontsize)
            # Target visualization
            axes[i, 0].imshow(np.moveaxis((image[i]*mask[i]).cpu().detach().numpy(), 0, -1))
            axes[i, 0].axis('off')

            axes[i, 1].imshow((actual_shading[i]*mask[i]).cpu().detach().numpy().squeeze(), cmap='gray')
            axes[i, 1].axis('off')

            axes[i, 2].imshow((mask[i].cpu().detach().numpy().squeeze()), cmap='gray')
            axes[i, 2].axis('off')

            # Predicted visualization
            axes[i, 3].imshow(np.moveaxis((appearance[i]*mask[i]).cpu().detach().numpy(), 0, -1))
            axes[i, 3].axis('off')

            axes[i, 4].imshow((pred_shading[i]*mask[i]).cpu().detach().numpy().squeeze(), cmap='gray')
            axes[i, 4].axis('off')

            axes[i, 5].imshow((pred_specular[i]*mask[i]).cpu().detach().numpy().squeeze(), cmap='gray')
            axes[i, 5].axis('off')

            # TEMP: Normalize to 0...1 when visualizing
            axes[i, 6].imshow(((fmel[i]*mask[i] - torch.min(fmel[i]*mask[i]))/(torch.max(fmel[i]*mask[i]) - torch.min(fmel[i]*mask[i]))*mask[i]).cpu().detach().numpy().squeeze(), cmap=cmap)
            axes[i, 6].axis('off')

            # TEMP: Normalize to 0...1 when visualizing
            axes[i, 7].imshow(((fblood[i]*mask[i] - torch.min(fblood[i]*mask[i]))/(torch.max(fblood[i]*mask[i]) - torch.min(fblood[i]*mask[i]))*mask[i]).cpu().detach().numpy().squeeze(), cmap=cmap)
            axes[i, 7].axis('off')

        plt.axis('off')
        plt.show(block=False)
        plt.pause(7)
        plt.close()

    def visualize_output(self, image, appearance, pred_shading, pred_specular, fmel, fblood, num=1, cmap='cividis'):
        """
        visualize actual input image and predicted outputs
        @input:
            same as model's outputs and targets
            num: number of samples to show (rows in plots, default=1)
            cmap: only for fmel & fblood, string of color map accepted by matplotlib (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
        @output:
            None, plotting 1x6 plots
        """
        # determine visualization size, recommend (8, 6) for local, (16, 12) for Colab
        fig_size = (8, 6)
        fontsize = 7
        if self.viz_big:
            fig_size = (16, 12) # really big
            fontsize = 9

        fig, axes = plt.subplots(num, 6, figsize=fig_size)
        # fig.suptitle("Sample {} images from decode output as training progress with order:\n\
        #     Target(first three), Predicted(the rest)\n\
        #         Input Image, Computed Shading, Mask | Pred Appearance, Pred Shading, Pred Spec, Pred Fmel, Pred Fblood".format(num))
        title_list = ['Input', 'Pred Face', 'Pred Shading', 'Pred Specular', 'fmel', 'fblood']
        for i in range(num):
            # add column header
            for col in range(6):
                axes[col].set_title(title_list[col], fontsize=fontsize)
            # Target visualization
            axes[0].imshow(np.moveaxis((image[i]).cpu().detach().numpy(), 0, -1))
            axes[0].axis('off')

            # Preted visualization
            axes[1].imshow(np.moveaxis((appearance[i]).cpu().detach().numpy(), 0, -1))
            axes[1].axis('off')

            axes[2].imshow((pred_shading[i]).cpu().detach().numpy().squeeze(), cmap='gray')
            axes[2].axis('off')

            axes[3].imshow((pred_specular[i]).cpu().detach().numpy().squeeze(), cmap='gray')
            axes[3].axis('off')

            # TEMNormalize to 0...1 when visualizing, no mask when predicting, so this normalization might be a little bit off for real face region
            axes[4].imshow(((fmel[i] - torch.min(fmel[i]))/(torch.max(fmel[i]) - torch.min(fmel[i]))).cpu().detach().numpy().squeeze(), cmap=cmap)
            axes[4].axis('off')

            # TEMNormalize to 0...1 when visualizing, no mask when predicting, so this normalization might be a little bit off for real face region
            axes[5].imshow(((fblood[i] - torch.min(fblood[i]))/(torch.max(fblood[i]) - torch.min(fblood[i]))).cpu().detach().numpy().squeeze(), cmap=cmap)
            axes[5].axis('off')

        plt.axis('off')
        plt.show()


# ONLY FOR TESTING
if __name__ == "__main__":
    test_image = torch.rand((1, 3, 64, 64))
    model = BioFaceNet()
    # Test forward pass
    # model(test_image)
    t = model.read_Tmatrix()
    print(t.shape)