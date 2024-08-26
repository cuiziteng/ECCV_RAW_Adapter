import numpy as np
import scipy
import scipy.stats as ss
import scipy.signal as sg
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import random
import cv2

def imshow(x, title=None, cbar=False, figsize=None, filename = 'kernel'):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.savefig(filename+'.jpg')

def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k

def isotropic_Gaussian(ksize=15, l=6):
    """ generate an isotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        l    : [0.1,50], scaling of eigenvalues
    Returns:
        k     : kernel
    """

    V = np.array([[1, 0], [0, -1]])
    D = np.array([[l, 0], [0, l]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k

def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.
    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k

def PCA(X, k=20):
        X = torch.from_numpy(X)
        X_mean = torch.mean(X, 0)
        X = X - X_mean.expand_as(X)

        v, w = torch.eig(torch.mm(X, torch.t(X)), eigenvectors=True)
        return torch.mm(w[:k, :], X).numpy()

if __name__ == "__main__":
    #design kernel
    kernel_size = int(random.choice([7, 9, 11, 13, 15, 17, 19, 21]))    # size
    kernel_width_iso = random.uniform(0.1, 2.4)
    angle = random.uniform(0, np.pi)
    kernel_width_un1 = random.uniform(0.5, 6)
    kernel_width_un2 = random.uniform(0.5, kernel_width_un1)

    k1 = isotropic_Gaussian(17, 2.4)
    
    # k2 = anisotropic_Gaussian(17, np.pi/3, 5, 2.3)
    # print(k2.shape)

    k3 = anisotropic_Gaussian(21, np.pi, 2, 4.5)
    plt.imsave('kernel3.jpg', k3, cmap='gray')