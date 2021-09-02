# BioFaceNet Pytorch Implementation
## Introduction
---
This is a pytorch implementation of the paper: [BioFaceNet: Deep Biophysical Face
Image Interpretation](https://arxiv.org/pdf/1908.10578.pdf). 


## How to Run (train/predict)
---
### Training: 
#### On local machine:
```Bash
# Clone this repo
git clone https://github.com/bznick98/BioFaceNet.git

# Prepare training data
"""
1. Data can be downloaded from google drive: https://drive.google.com/drive/folders/1UMiaw36z2E1F-tUBSMKNAjpx0o2TePvF
2. Training data are in pairs, for 1 set of training data, you need to download 2 hdf5 files. 
3. For instance, 'zx_7_d10_inmc_celebA_20.hdf5' has 2533 samples. To train on these 2533 samples, you also need to download the corresponding lighting parameters: 'zx_7_d3_lrgb_celebA_20.hdf5'. If you are trying to do a demo, then only data ending with number '20' and '05' are recommend to download. '20' and '05' has total about 12k samples.
4. After downloaded the data, you need to put them in directory data/, where data/ should be in the project root directory.
"""

# [Optional] Specify which data to train in train.py, by default will train 2533 samples in 'zx_7_d10_inmc_celebA_20.hdf5'
"""
Modify inmc_list and lrgb_list in function train() in train.py
"""

# [Optional] Set hyperparameters if not want to use default setting
"""
Hyperparameters can be modified by calling train.py with different arguments, see argparse part of train.py or executing 'python train.py -h' for detail.
"""

# Training
python train.py  # [with optional args]

# Models will be saved at checkpoints/ every epoch
```

#### On Google Colab:
```Bash
# Open BioFaceNet.ipynb using Google Colab
"""
BioFaceNet.ipynb is essentially a modified train.py, all the other related code will be pulled from github repo.
"""

# Prepare training data
"""
1. Data is stored in the Google Drive shared by Zhixin Shu, one of the authors of Neural Face Editing paper. 
2. To use the data, first open the shared google drive: https://drive.google.com/drive/folders/1UMiaw36z2E1F-tUBSMKNAjpx0o2TePvF
3. Click the toggle-down menu in the folder's title, select 'Add shortcut to Drive'.
4. Then you can access this folder in Colab by mounting your own google drive to Colab.
"""
```
![Adding-shortcut-to-your-own-gdrive](readme-imgs/shortcut.png)
```Bash
# Execute each cells or Run All.
# Models will be saved at /content/BioFaceNet/checkpoints/
```