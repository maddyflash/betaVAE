"""
This is the file with all of the functions for using the vae with dsprites.

the functions are:
 - Dsprites to import data
 - imshow to show images
 - make_grid to make a grid

Reorganized with very slight modifications from code in "harry" or master branch

Note: as mentioned, a significant portion of the dsprites class for using the dsprites dataset was taken from https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
"""


"""
# import statements
import tensorflow as tf
import sys
"""

import matplotlib.pyplot as plt
import numpy as np
import typing
import math
import tqdm
import os
import itertools
import uuid
import copy
import subprocess
import imageio
import IPython


from IPython import display
from IPython.display import Image
from pathlib import Path
from enum import IntEnum


"""
# system check functions
def test_tf():
    ''' tests that tf and gpu are loaded properly'''
    print(sys.version)
    print(tf.__version__)
    print(tf.test.is_gpu_available())
    
"""


# ------------------------------------------------------------------------------ #
# Dsprites class functions
class DSprites:
    ''' class to interact with the dsprites data set'''
    class Latents(IntEnum): #these are the ground truth latents
        '''class accesses ground truth latents'''
        COLOUR, SHAPE, SCALE, ORIENTATION, XPOS, YPOS = range(6)
        
    """
    A significant portion of this class is taken as-is from the manual
    https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
    """
    
    def __init__(self, path=".", download=False): 
        """ Inits Dsprites class (ie loads data)
        
        :param path: (str)= ".": file path to data
        :param download: (bool)=False: ????? TODO question
        
        :return: none; loads data
        
        """
        self._filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        path = Path(path).absolute()
        assert path.exists()
        data = path / self._filename
        if not data.exists():
            if download:
                subprocess.run(["wget", "-O", str(data),
                                "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"])
            else:
                raise ValueException("Can't find dataset, use download=True to download.")
        data = np.load(str(data), encoding='bytes', allow_pickle=True)

        # ====

        imgs = data['imgs'] #(737280 imgs each of dim 64x64, uint8)
        metadata_raw = data['metadata'][()] #random info inc possible latent values
        self._metadata = dict()
        for k, v in metadata_raw.items():
            self._metadata[k.decode()] = v

        # NOTE: can't cast now because our notebook runs out of RAM. cast in map instead
        #imgs = imgs.reshape(-1, 64, 64, 1).astype(np.float32) #try uncommenting
        # was able to uncomment this, so no need to cast prior to model in map
        self._imgs = imgs.reshape(-1, 64, 64, 1)

        # for example: array([ 0,  0,  2, 37, 15, 22])
        # i.e. the relative (normalised) change in latent factors
        # bc images were varied one latent at a time?
        self._latents_classes = data['latents_classes'] 
        #(737280 ims x6 ints representing int index of latent factor values)

        # for example: array([1., 1., 0.7 , 5.96097068, 0.48387097, 0.70967742])
        # i.e. the actual latent values used to generate the image
        self._latents_values = data['latents_values']
        #(737280 ims x6 floats representing actual latent factor values)

        # specification: the number of varying "degrees" of change in each
        # dimension corresponding to an independent generative factor
        # array([ 1,        3,      6,           40,  32,  32])
        #       colour, shape,  scale,  orientation,   X,   Y
        self._latents_sizes = self._metadata['latents_sizes']
        # ??? about how this differs from latents_classes

        # for easy conversion from latent vector to indices later (see latent_to_idx)
        # essentially: array([737280, 245760,  40960,   1024,     32,      1])
        self._latents_bases = np.r_[self._latents_sizes[::-1].cumprod()[::-1][1:], 1]
        # ??? again question here
    
    def latent_size(self, latent: 'DSprites.Latents') -> int:
        """
        :param latent: of type DSprites.Latents (an enum class)
        :return: the maximum integer allowed for the specified `latent`
        """
        return self._latents_sizes[latent.value]
    
    def to_idx(self, latents: np.array) -> int:
        """
        convert latent vector into index that can then be used to index
        the actual image in self._imgs
        """
        return np.dot(latents, self._latents_bases).astype(int)
    
    def sample_latent(self, n: int=1, fixed: 'DSprites.Latents'=None) -> np.array:
        """
        randomly samples `n` latent vectors

        :param n: number samples
        :param fixed: if not `None`, then in all samples, this latent is kept
                     fixed based on a random draw. The rest of the latents are
                     random.
        :return: an `np.array` of shape nx6
        """
        samples = np.zeros((n, self._latents_sizes.shape[0]))
        for i, lat_size in enumerate(self._latents_sizes):
            samples[:, i] = np.random.randint(lat_size, size=n)
        if fixed:
            samples[:, fixed] = np.random.randint(0, ds.latent_size(fixed))
        return samples
    
    @property
    def imgs(self) -> np.array:
        return self._imgs

    def subset(self, size=50_000) -> np.array:
        """
        returns a subset of the images. (Workaround for memory constraints)
        :param size: number of samples to return
        """
        return self._imgs[np.random.choice(self._imgs.shape[0], size=size, replace=False)]


# ------------------------------------------------------------------------------ #
# data management functions
def make_grid(tensor: np.array, nrow: int=8, padding: int=2, pad_value: int=0) -> np.array:
    """
    adapted from: https://pytorch.org/docs/stable/_modules/torchvision/utils.html#make_grid
    :param tensor: nxwxhxc np.array
    :param nrow: number of rows to use.
    :param padding: padding between images
    :param pad_value: value used to pad
    :return: np.array of dimension 3 (WxHxC) with all images arranged in a grid.

    """
    if tensor.shape[0] == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    num_channels = tensor.shape[3]
    grid = np.full((height * ymaps + padding, width * xmaps + padding, num_channels), pad_value, dtype=tensor.dtype)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps: break
            ystart = y * height + padding
            xstart = x * width + padding
            grid[ystart:(ystart + height - padding), ...][:, xstart:(xstart + width - padding), :] = tensor[k]
            k = k + 1
    return grid.squeeze()

def imshow(img: np.array, title: str='', ax: plt.Axes=None):
    if not ax:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.set_xticks(())
    ax.set_yticks(())
    if title:
        ax.set_title(title)

# ------------------------------------------------------------------------------ #
# main function
if __name__ == "__main__":
    ''' main function'''
    print("hello world; this is myvae.py")
