"""
camera model described in the paper
"""
import re
import numpy as np
import scipy.io

from sklearn.decomposition import PCA


"""
Camera Sensitivity Database Comes From:
- 3 x 33 x 28 camera sensitivity database, by Jinwei Gu
http://www.gujinwei.org/research/camspec/camspec_database.txt
- Seems like also on Github Matlab implementation: util/rgbCMF.mat
https://github.com/ssma502/BioFaces/tree/master/util
"""

def read_camspec(filepath="utils/camspec_database.txt"):
    """
    parse camera sensitivity database, transform into np.array
    @input:
        filepath: the text file contains 28 camera sensitivity measures
    @output:
        rgbCMF: a 3x33x28 matrix that contains 28 camera sensitivity measures
                    each for 33 discretized bandwiths and 3 rgb channels
    """
    with open(filepath) as f:
        lines = f.readlines()
    
    # remove camera names in the list, then trans from string to float
    lines = [i for i in lines if i[0].isdigit()]
    lines = ''.join(lines)
    lines = re.split('; |\t|\n', lines)
    lines = [np.float64(i) for i in lines]
    lines = np.array(lines)
    # split into group of 33 first (wavelength values)
    lines = np.array(np.split(lines, len(lines)/33))
    # split into group of 3 rgb
    lines = np.array(np.split(lines, len(lines)/3))
    # result in a (28, 3, 33) ndarray, transpose it to (3, 33, 28)
    rgbCMF = np.transpose(lines, (1, 2, 0))

    # it's separated by r, g, b (3); then wavelengths (33); then camera models (28)
    return rgbCMF


def camera_PCA(rgbCMF):
    """
    performs PCA on camera sensitivity database
    @input:
        rgbCMF: a camera sensitivity database measured by Jinwei Gu et,al.
                    a matrix with shape 3 x 33 x 28, where 3 represents
                    R, G, B channels, 33 represents 33 discretized bandwidths
                    from 400nm to 720nm, 28 represents 28 camera models measured
    @ouput:
        PC, mean, eigenvalues: the first two PCA components and relative results
    """
    # read 3x33x28
    # rgbCMF = read_camspec()
    # print(rgbCMF.shape)

    # 3x33x28 was squeezed to 99x28 in Matlab implementation
    redS = rgbCMF[0]
    greenS = rgbCMF[1]
    blueS = rgbCMF[2]

    X = np.zeros((99, 28))
    for i in range(28):
        X[:33, i] = redS[:, i] / np.sum(redS[:, i])
        X[33:66, i] = greenS[:, i] / np.sum(greenS[:, i])
        X[66:99, i] = blueS[:, i] / np.sum(blueS[:, i])

    X = X.T # (28, 99)

    # center data
    mean_feat = X.mean(axis=0) #(99, )
    X -= mean_feat # (28, 99)

    pca = PCA(n_components=2)
    pca.fit(X)
    components = pca.components_ # (2, 99)
    eigenvalues = pca.explained_variance_
    # print(pca.explained_variance_ratio_)
    # print(eigenvalues)
    mean = pca.mean_

    # get first two principle components
    PC = np.matmul(components[:2, :].T, np.diag(np.sqrt(eigenvalues)))

    # print(PC.shape, eigenvalues.shape, mean.shape)

    return PC, mean, eigenvalues

def camera_model(mu, PC, b, wavelength=33):
    """
    return e, Sr, Sg, Sb, resembles cameraModel func in Matlab
    @input:
        mu: mean value across features(wavelengths)
        PC: principle components (first two)
        b: camera parameters (2-dimensional vec)
        wavelength: how many discretized wavelengths, default=33
    @output:
        e:
        Sr, Sg, Sb:
    """
    # vec(S(b)) = PC * diag(eigenvalues1, ...) + vec(mean(S))
    S = np.matmul(PC, np.diag(b)) + mu

    # apply ReLU (keep positive part)
    S[S < 0] = 0 # (Batchsize, 99)

    # split S into Sr, Sg, Sb
    Sr = S[:, :wavelength]
    Sg = S[:, wavelength:wavelength*2]
    Sb = S[:, wavelength*2:wavelength*3]

    return Sr, Sg, Sb

# ONLY FOR TESTING
if __name__ == '__main__':
    rgbCMF = read_camspec()
    camera_PCA(rgbCMF)