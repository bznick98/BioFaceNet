import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset


class CelebADataset(Dataset):
    """
    The original CelebA dataset does not have [masks/spherical harmonics parameters], instead it's 
    probably from an augmented version mentioned in paper Neural Face Editing, which links to its data 
    at: https://drive.google.com/drive/folders/1UMiaw36z2E1F-tUBSMKNAjpx0o2TePvF
    2 hdf5 files need to be downloaded in pair (for 1 set of data), which is:
        - zx_7_d10_inmc_celebA_**.hdf5
        - zx_7_d3_lrgb_celebA_**.hdf5
    where inmc has rgb, normal, mask, spherical harmonics parameters for images and lrgb has light parameters for images
    """
    def __init__(self, inmc_list, lrgb_list):
        """
        read inmc and lrgb data
        @input:
            inmc_list: a list of filepaths for inmc hdf5 files
            lrgb_list: a list of filepaths for lrgb hdf5 files, need to match the order of inmc file list
        """
        # preprocessing functions
        self.trans_3d = transforms.Compose([
            # transforms.ToTensor(), # to tensor, normalized to [0, 1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # normalized to [-1, 1]
        ])
        self.trans_1d = transforms.Compose([
            # transforms.ToTensor(), # to tensor (will convert HxWxC to CxHxW)
            # transforms.Normalize(mean=(0.5,), std=(0.5,)) # normalized to [-1, 1]
        ])

        # reading inmc hdf5
        for i, fp in enumerate(inmc_list):
            self.inmc_hdf5 = h5py.File(fp, 'r')
            if i == 0:
                self.imgs = np.array(self.inmc_hdf5['zx_7'])
            else:
                self.imgs = np.concatenate((self.imgs, self.inmc_hdf5['zx_7']))
        self.len = self.imgs.shape[0]

        # reading lrgb hdf5
        for i, fp in enumerate(lrgb_list):
            self.lrgb_hdf5 = h5py.File(fp, 'r')
            if i == 0:
                self.lrgbs = self.lrgb_hdf5['zx_7']
            else:
                self.lrgbs = np.concatenate((self.lrgbs, self.lrgb_hdf5['zx_7']))

        # number of inmc data should equal that of lrgb data
        assert self.len == self.lrgbs.shape[0]
        print("{} data samples are loaded.".format(self.len))

    def shading(self, L, normal):
        """
        the dataset does not have shading, compute shading
        from spherical harmonics coefficient based on the paper
        "Neural Face Editing with Intrinsic Image Disentangling"
        (https://arxiv.org/pdf/1704.04131.pdf), eq.4 & eq.5
        @input:
            L: lighting, 9-dimensional spherical harmonics coefficent
                (9-dim vector from lrgb hdf5 files)
            normal: normal map from inmc hdf5 files
        @output:
            shading: 3xHxW
        """
        # c1, c2, c3, c4, c5 in eq.5
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        # compute 3 channel separately
        S = []
        for ch in range(3):
            # normal matrix -> 3x64x64
            # [n_i; 1] -> 4x64x64 (added 1-dim of ones)
            # print("NORMAL SHAPE: ", normal.shape)
            normal_1 = np.concatenate((normal, np.ones((1, normal.shape[-2], normal.shape[-1]))))
            normal_1_T = np.moveaxis(normal_1, 0, -1)

            # print("AA: ", normal_1.shape)

            # K Matrix
            K = np.array([
                [c1*L[ch,8], c1*L[ch,4], c1*L[ch,7], c2*L[ch,3]],
                [c1*L[ch,4], -c1*L[ch,8], c1*L[ch,5], c2*L[ch,1]],
                [c1*L[ch,7], c1*L[ch,5], c3*L[ch,6], c2*L[ch,2]],
                [c2*L[ch,3], c2*L[ch,1], c2*L[ch,2], c4*L[ch,0]-c5*L[ch,6]]
            ])

            # Shading S (it seems original paper implement same shading across 3 channels)
            S.append(np.sum(np.matmul(normal_1_T, K) * normal_1_T, axis=-1)[None, ...])

        # stack 3 channels
        shading = np.vstack(S)
        return shading

    
    def __getitem__(self, index):
        """
        5 items per dataset sample, with order:
            rgb_img, 
            xyz_normal,
            mask,
            shading,
            spherical_harmonics_param (not used rn)
        @output:
            image: 3xHxW
            normal: 3xHxW
            mask: 1xHxW
            shading: 1xHxW, torch.Tensor, normalized to
            spherical_harmonics_param: 3xHxW
        """
        # all these ndarray loaded are already in range 0...1, CxHxW
        #   only need to convert them to torch.Tensor type
        image = self.imgs[index][0:3] # 3xHxW, ndarray
        normal = self.imgs[index][3:6] # 3xHxW, ndarray
        mask = self.imgs[index][6] # HxW, ndarray
        spherical_harmonics_param = self.imgs[index][7:10] # 3xHxW, ndarray

        # print("LOADED image MAX/MIN: ", np.max(image), np.min(image))
        # print("LOADED image SHAPE: ", image.shape)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        # lighting
        L = self.lrgbs[index]

        # calculate shading from illumination(Lighting) and geometry(Normal)
        shading = self.shading(L, normal)

        # print("COMPUTED shading MAX/MIN, shape: ", np.max(shading), np.min(shading), shading.shape)

        # ndarray: (0...1, CxHxW) to Tensor: (0...1, CxHxW)
        image = torch.Tensor(image)
        normal = torch.Tensor(normal)
        mask = torch.Tensor(mask)
        spherical_harmonics_param = torch.Tensor(spherical_harmonics_param)
        shading = torch.Tensor(shading)

        # print("Tensor-fy shading MAX/MIN: ", torch.max(shading), torch.min(shading))
        # print("Tensor-fy shading SHAPE: ", shading.shape)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        # convert shading from 3-channel to 1-channel(grayscale)
        to_grayscale = transforms.Compose([
            transforms.Grayscale(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)) # normalized to [-1, 1]
        ])
        shading = to_grayscale(shading).float()

        # Tensor normalize to -1...1 except for mask: CxHxW
        image = self.trans_3d(image).float()
        normal = self.trans_3d(normal).float()
        mask = self.trans_1d(mask[None, ...]).float() # [0, 1]
        spherical_harmonics_param = self.trans_3d(spherical_harmonics_param).float()
        # print(type(image), type(normal), type(mask), type(shading), type(spherical_harmonics_param))


        # print("Transformed IMG MAX/MIN: ", torch.max(image), torch.min(image))
        # print("Transformed IMG SHAPE: ", image.shape)
        # print("^^^^^^^^^^^--------^^^^^^^^^^^^^^")
        # print("Transformed shading MAX/MIN: ", torch.max(shading), torch.min(shading))
        # print("Transformed shading SHAPE: ", shading.shape)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        # print("GETITEM SHAPE: \n", image.shape, normal.shape, mask.shape, shading.shape, spherical_harmonics_param.shape)
        return image, normal, mask, shading, spherical_harmonics_param

    def __len__(self):
        return self.len


class CelebADataLoader:

    def __init__(self, inmc_list, lrgb_list):
        """
        initialize dataloader for celebA dataset (NeuralFaceEditing augmented dataset)
        @input:
            inmc_list: a list of filepaths for inmc hdf5 files
            lrgb_list: a list of filepaths for lrgb hdf5 files, need to match the order of inmc file list
        """
        dataset = CelebADataset(inmc_list, lrgb_list)

        self.loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4
        )

    def show_sample(self, num=5):
        """
        show 5 sample from celebA dataset
        @input:
            num: number of samples to plot, default=5
        @output:
            None
        """
        fig, axes = plt.subplots(num, 5)
        fig.suptitle("Sample {} images from CelebA dataset".format(num))
        it = iter(self.loader)
        for i in range(num):
            image, normal, mask, shading, spherical_harmonics_param = next(it)

            print("SHOW Max/Min: ", torch.max(image), torch.min(image))

            # print(image[0].shape, normal[0].shape, mask[0].shape, spherical_harmonics_param.shape)
            axes[i, 0].imshow((image[0].permute(1,2,0)))
            axes[i, 0].axis('off')

            axes[i, 1].imshow((normal[0].permute(1,2,0)))
            axes[i, 1].axis('off')

            axes[i, 2].imshow((torch.squeeze(mask[0])), cmap='gray')
            axes[i, 2].axis('off')

            axes[i, 3].imshow((shading[0].permute(1,2,0)), cmap='gray')
            axes[i, 3].axis('off')

            axes[i, 4].imshow((spherical_harmonics_param[0].permute(1,2,0)))
            axes[i, 4].axis('off')
        
        plt.axis('off')
        plt.show()



if __name__ == '__main__':
    inmc_list = [
        'data/zx_7_d10_inmc_celebA_20.hdf5',
        # 'data/zx_7_d10_inmc_celebA_05.hdf5',
    ]
    lrgb_list = [
        'data/zx_7_d3_lrgb_celebA_20.hdf5',
        # 'data/zx_7_d3_lrgb_celebA_05.hdf5',
    ]
    CelebADataLoader(inmc_list, lrgb_list).show_sample() 