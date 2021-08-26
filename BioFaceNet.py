from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F


class BioFaceNet(nn.Module):
    def __init__(self):
        """
        basically a UNet plus fully connected branch from the lowest resolution
        """
        super().__init__()

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
        self.t_conv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)
        self.t_conv3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2)
        self.t_conv4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2)

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
        #   for predicting vector quantities (b, wA, wD, t, wF1, ...)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 256),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(128), 
            nn.Linear(128, 17), # bSize(2) + LightVectorSize(15)
        )

    def forward(self, image):
        """
        forward passing
        @input:
            image: a batch of rgb images
        @output:
            fmel: melanin map
            fblood: haemoglobin map
            shading: diffuse? shading map
            specular: specular shading map
            b: 2-dim vector of camera sensitivity parameters
            lighting_params: 15-dim vector of lighting parameters
        """
        print("Input image size: ", image.shape)
        # encoder
        x1 = self.down1(image) #
        x2 = self.maxpool_2x2(x1)

        print("Size after 1: ", x2.shape)

        x3 = self.down2(x2) #
        x4 = self.maxpool_2x2(x3)

        print("Size after 2: ", x4.shape)

        x5 = self.down3(x4) #
        x6 = self.maxpool_2x2(x5)

        print("Size after 3: ", x6.shape)

        x7 = self.down4(x6) # 
        x8 = self.maxpool_2x2(x7)

        print("Size after 4: ", x8.shape)

        x9 = self.down5(x8)

        # lowest resolution output (1, 512, 4, 4) if input is 64x64
        low_res = x9

        print("Size of lowest resolution: ", x9.shape)

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
        print(x.shape)

        # x.shape = (B, 4, W, H)
        fmel = x[:, 0, :, :]
        fblood = x[:, 1, :, :]
        shading = x[:, 2, :, :]
        specular = x[:, 3, :, :]

        # fully connected branch from the lowest resolution
        vector_output = self.fc(low_res)
        print(vector_output.shape)
        b = vector_output[:2]
        lighting_params = vector_output[2:]

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

    def reconstruct_face():
        """
        decoding from physical-based parameters
        @input:
        
        @output:
            appearance: reconstructed face raw image from parameters
        """

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
        @output:
            weightA: used for illum. model
            weightD: used for illum. model
            Fweights: used for illum. model
            CCT: used for illum. model, color temperature, bounded [1,22]
            b: camera sensitivity params, bounded [-3,3]
            BGrid: X
            fmel: scaled melanin map, bounded [-1,1]
            fblood: scaled haemoglobin map, bounded [-1,1]
            shading: scaled shading, exponentiated
            spec: scaled specularity, exponentiated
        """
        # Normalize lighting parameters (wA, wD, wF1~wF12) so that sum to 1
        lighting_params = F.softmax(lighting_params)
        weightA = lighting_params[0]
        weightB = lighting_params[1]
        CCT = lighting_params[2]
        CCT = ((22 - 1) / (1 + torch.exp(-CCT))) + 1
        Fweights = lighting_params[3:]
        b = 6 * F.sigmoid(b) - 3            # [-3, 3]
        fmel = F.sigmoid(fmel) * 2 - 1      # [-1, 1]
        fblood = F.sigmoid(fblood) * 2 - 1  # [-1, 1]
        shading = torch.exp(shading)        # must positive
        spec = torch.exp(spec)              # must positive

        return weightA, weightB, CCT, Fweights, b, fmel, fblood, shading, spec



# ONLY FOR TESTING
if __name__ == "__main__":
    test_image = torch.rand((1, 3, 64, 64))
    model = BioFaceNet()
    # Test forward pass
    model(test_image)