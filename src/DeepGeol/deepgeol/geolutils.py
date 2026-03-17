# coding: utf-8 -*-
#
# This file is part of DeepGeol.
#
# DeepGeol is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# DeepGeol is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DeepGeol.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2024-2025 Anthony Larcher
"""


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2024-2025 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = "reS"

from random import random

import numpy
import os
import pandas
import rasterio
import torch

from rasterio.windows import Window
from torchvision.transforms import v2
from torch.utils.data import Dataset





def remove_nan(image):
    """
    Function that replace Nan in a Numpy.array by the mean of all other values
    :param image: numpy.array the array to clean
    :return: the clean numpy.arrayt without Nan
    """
    image[numpy.isnan(image)] = numpy.nanmean(image)
    return image

def clean_mask(mask):
    """
    Function that clean mask by removing values that are unproper and replacing them by 0
    :param mask: the input mask
    :return: the cleaned mask
    """
    mask[mask == -9999] = 0
    return mask

def normalize_data(data):
    """
    Normalize data by subtract the min value and divide by the whole range (max-min)
    :param data: the input data to normalize
    :return: a normalized version of the input
    """
    return (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))


def _split_image(filename, height, width, train_val_dev=(1, 1, 1), crop_mode='vertical'):
    """
    Take an image as input and split it according to the ratios of train_val_dev
    :param filemane: str name of the original file
    :param height: int height of the image, in number of pixels
    :param width: int width of the image, in number of pixels
    :param train_val_dev: a triplet with numbers that reflect the quantity of sub-windows to take into each set
    :param crop_mode: cut the image vertically or horizontally
    """
    stride = dict()
    stride['filename'] = filename
    stride['train'] = dict()
    stride['validation'] = dict()
    stride['dev'] = dict()

    train, validation, dev = tuple(numpy.array(train_val_dev) / numpy.array(train_val_dev).sum())
    if crop_mode == 'vertical':
        stride['train']['h_stride'] = 0
        stride['train']['w_stride'] = 0
        stride['train']['height'] = numpy.fix(height * train).astype(int)
        stride['train']['width'] = width

        stride['validation']['h_stride'] = numpy.fix(height * (train)).astype(int)
        stride['validation']['w_stride'] = 0
        stride['validation']['height'] = numpy.fix(height * (train + validation)).astype(int) - numpy.fix(height * (train)).astype(int)
        stride['validation']['width'] = width

        stride['dev']['h_stride'] = numpy.fix(height * (train + validation)).astype(int)
        stride['dev']['w_stride'] = 0
        stride['dev']['height'] = height - numpy.fix(height * (train + validation)).astype(int)
        stride['dev']['width'] = width

    elif crop_mode == 'horizontal':
        stride['train']['h_stride'] = 0
        stride['train']['w_stride'] = 0
        stride['train']['height'] = height
        stride['train']['width'] = numpy.fix(width * train).astype(int)

        stride['validation']['h_stride'] = 0
        stride['validation']['w_stride'] = numpy.fix(width * (train)).astype(int)
        stride['validation']['height'] = height
        stride['validation']['width'] = numpy.fix(width * (train + validation)).astype(int) - numpy.fix(width * (train)).astype(int)

        stride['dev']['h_stride'] = 0
        stride['dev']['w_stride'] = numpy.fix(width * (train + validation)).astype(int)
        stride['dev']['height'] = height
        stride['dev']['width'] = width - numpy.fix(width * (train + validation)).astype(int)
    else:
        print('WRONG crop_mode')

    return stride


def _crop_split(filename,
                height,
                width,
                stride_h,
                stride_w,
                window_size=(512, 512),
                stride=(256, 256)
               ):
    """
    Given a part of an image (split) with its height and width, compute all possible sub-windows
    that could be extracted from this split for the given window size and stride
    This function returns a list of tuples with (filename, h_stride, w_stride)
    :param filename: str name of the file to split
    :param height: int height of the original image to crop,
    :param width: int  width of the original image to crop,
    :param stride_h: int stride in the height dimension (dimension 0)
    :param stride_w: int stride in the width dimension (dimension 1)
    :param window_size: tuple(int, int) size of the crops to produce, default is (512, 512)
    :param stride: tuple(int, int) stride of the croping, default is (256, 256)
    """
    window_list = []

    # Compute the number of possible windows given the window size and the stride
    patch_h_out = numpy.fix(((height - (window_size[0] - 1) - 1)/stride[0]) + 1).astype(int)
    patch_w_out = numpy.fix(((width - (window_size[1] - 1) - 1)/stride[1]) + 1).astype(int)

    # Create the list of coordonates for each window in the image
    h = numpy.arange(patch_h_out) * stride[0]
    w = numpy.arange(patch_w_out) * stride[1]

    xv, yv = numpy.meshgrid(h, w, indexing='ij')
    coord = numpy.concatenate((xv[None, ...], yv[None, ...]), axis=0)
    coord = numpy.transpose(coord, axes=(1, 2, 0))
    coord = coord.reshape(-1, 2).astype(numpy.uint16)

    window_list += [(filename, xy[0] + stride_h, xy[1] + stride_w)for xy in coord]

    return window_list


def get_crop_list(data_path,
                  mask_path,
                  window_size=(512, 512),
                  stride=(256, 256),
                  train_val_dev=(0.8, 0.2, 0.0),
                  splitting_mode='default',
                  crop_mode='vertical'):
    """
    Function that get the images and masks from two directories
    and create the list of sub-windows (crop) of the image
    Different splitting modes are available to split the data between Train, Validation and Dev

    :param data_path: str directory where images are stored
    :param mask_path: str directory where masks are stored
    :param splitting_mode: different ways to split the data between train/val/dev
        - default: random split across all possible sub-windows, this is the most straightforward mode but
            it can create biases due to the overlap of the windows, parts of the validation and development sub-windows can be
            partly overlaping across the three sets
        - safe_images: in this mode, each image is fully allocated to train,validation or dev set
            images are allocated to each set according to their size in order to be as closed as possible from the required ratios
        - safe_split: in this case, each image is first split into non-overlapping crops that are then allocated to the three sets
            in this mode, each set (train/valid/dev) includes one part of each image.
            In this mode, the final number of sub-windows might be less that in default mode as it forbids some sub-windows overlap
        - crop_mode: only for 'safe_split' mode, determine the dimension according to which each image is divided
    :return: a Pandas.DataFrame with all crops from the dataset
    """
    raw_images_list = []
    problem_images = []

    # Compute percentage of data from train_val_dev
    train_val_dev = numpy.array(train_val_dev) / numpy.array(train_val_dev).sum()

    # Get the list of images, for each of them, check that the corresponding mask is available and that dimensions are similar
    file_list = [f for f in os.listdir(data_path) if f.endswith('.tif')]
    for filename in file_list:
        # Convert image name to mask name: '26_15_10m_v4.1_dem.tif' → 'dem_26_15.tif'
        parts = filename.split('_')
        mask_filename = f"dem_{parts[0]}_{parts[1]}.tif"
        mask_fullpath = os.path.join(mask_path, mask_filename)

        # Skip images that have no corresponding mask
        if not os.path.exists(mask_fullpath):
            continue

        with rasterio.open(os.path.join(data_path, filename)) as src:
            im_height = src.height
            im_width = src.width

        with rasterio.open(mask_fullpath) as src:
            mask_height = src.height
            mask_width = src.width

        if im_height == mask_height and im_width == mask_width:
            raw_images_list.append((filename, im_height, im_width))
        else:
            problem_images.append(filename)

    """
    For Default mode, divide each image in sub-windows and store the information in a DataFrame
    """
    training_windows = []
    validation_windows = []
    dev_windows = []

    if splitting_mode == 'default':
        sub_windows = []
        for image_fn, height, width in raw_images_list:
            sub_windows.extend(_crop_split(image_fn,
                                           height,
                                           width,
                                           0,
                                           0,
                                           window_size=window_size,
                                           stride=stride
                                          )
                              )
        # Randomly split the list of sub-windows into 3 parts according to the ratios
        tmp_df = pandas.DataFrame(sub_windows, columns=('filename', 'height', 'width'))
        idx = numpy.arange(len(tmp_df))
        random.shuffle(idx)
        training_df = tmp_df.iloc[idx[:int(len(tmp_df) * train_val_dev[0])]]
        validation_df = tmp_df.iloc[idx[int(len(tmp_df) * train_val_dev[0]): int(len(tmp_df) * (train_val_dev[0] + train_val_dev[1]))]]
        dev_df = tmp_df.iloc[idx[int(len(tmp_df) * (train_val_dev[0] + train_val_dev[1])):]]

    elif splitting_mode == 'safe_split':
        for image_fn, height, width in raw_images_list:
            splits = _split_image(image_fn, height, width, train_val_dev=train_val_dev, crop_mode=crop_mode)

            wl_train = _crop_split(splits['filename'],
                                   splits['train']['height'],
                                   splits['train']['width'],
                                   splits['train']['h_stride'],
                                   splits['train']['w_stride'],
                                   window_size=window_size,
                                   stride=stride
                                  )
            training_windows.extend(wl_train)

            wl_validation = _crop_split(splits['filename'],
                                        splits['validation']['height'],
                                        splits['validation']['width'],
                                        splits['validation']['h_stride'],
                                        splits['validation']['w_stride'],
                                   window_size=window_size,
                                   stride=stride
                                       )
            validation_windows.extend(wl_validation)

            wl_dev = _crop_split(splits['filename'],
                                 splits['dev']['height'],
                                 splits['dev']['width'],
                                 splits['dev']['h_stride'],
                                 splits['dev']['w_stride'],
                                   window_size=window_size,
                                   stride=stride
                                )
            dev_windows.extend(wl_dev)

        training_df = pandas.DataFrame(training_windows, columns=('filename', 'height', 'width'))
        validation_df = pandas.DataFrame(validation_windows, columns=('filename', 'height', 'width'))
        dev_df = pandas.DataFrame(dev_windows, columns=('filename', 'height', 'width'))

    elif splitting_mode == 'safe_images':
        # todo
        pass

    print(f'Split {len(training_df) + len(validation_df) + len(dev_df)} into {len(training_df)} / {len(validation_df)} / {len(dev_df)}')
    return training_df, validation_df, dev_df


class GeoSet(Dataset):
    """
    DataSet used to load crops of images associated with their binary mask
    """
    def __init__(
        self,
        data_df,
        data_path,
        mask_path,
        window_size
    ):
        """
        :param data_df: Pandas.DataFrame the decsription of all the crops to take
        :param data_path: str the directory to load images from
        :param mask_path: str the directory to load masks from
        """
        # Get the list of images and masks (assuming they are the same
        self.data_df = data_df
        self.data_path = data_path
        self.mask_path = mask_path
        self.window_size = window_size

        # Get the length of the dataset
        self.len = len(self.data_df)

    def __getitem__(self, index):

        filename = self.data_df.at[index, 'filename']
        x_origin = self.data_df.at[index, 'width']
        y_origin = self.data_df.at[index, 'height']

        #print(f'x_origin = {x_origin}')
        #print(f'y_origin = {y_origin}')

        # Open the source image corresponding and read only a window
        with rasterio.open(os.path.join(self.data_path, filename)) as src:
            window = src.read(1, window=Window(x_origin, y_origin, self.window_size, self.window_size))
            window = remove_nan(window)
            window = normalize_data(window)
            window = torch.from_numpy(window)[None, ...]

        # Open the source mask corresponding and read only a window
        parts = filename.split('_')
        mask_filename = f"dem_{parts[0]}_{parts[1]}.tif"
        with rasterio.open(os.path.join(self.mask_path, mask_filename)) as src:
            mask = src.read(1, window=Window(x_origin, y_origin, self.window_size, self.window_size))
            mask = clean_mask(mask)
            mask = torch.from_numpy(mask)[None, ...]

        transformations_list = [
            v2.RandomResizedCrop(size=(self.window_size, self.window_size), antialias=True),
            v2.RandomPhotometricDistort(p=2),
            v2.RandomHorizontalFlip(p=1),
            v2.RandomRotation(180),
            v2.GaussianNoise()
            ]
        # Pass the list of transformations to Compose function
        transformations = v2.RandomApply(transformations_list, p=0.7)
        window, mask = transformations(window, mask)
        return window, mask

    def __len__(self):
        """

        :param self:
        :return:
        """
        return self.len


def c(pred, target):
    """
    Assuming that pred and target are binary images (or list of aligned images)
    with positive values at 1 and negtive values at 0
    Compute:
        - TP: true positive rate
        - FP: false positive rate
        - TN: true negative rate
        - FN: false negative rate
    to then compute different metrics
    :param pred: an image predicted by the Segmentation Network of a list of images
    :param target: the reference mask to predict or a list of masks
    """

    # If pred and target are lists of imagesof same dimensions, 
    # convert them to a 3D-arrays
    if isinstance(pred, list) and isinstance(target, list):
        _pred = numpy.stack(pred, axis=0)
        _target = numpy.stack(target, axis=0)
    else:
        _pred = pred
        _target = target

    assert _pred.shape == _target.shape 

    # Convert images to boolean 2D-arrays
    _pred = _pred.astype('bool')
    _target = _target.astype('bool')

    # Compute true and false positive and negative rates
    ps = _pred.sum() if not _pred.sum() == 0 else 1
    nps = (~_pred).sum() if not (~_pred).sum() == 0 else 1
        

    true_positive = (_pred & _target).sum() / ps
    false_positive = (_pred & ~_target).sum() / ps
    true_negative = (~_pred & ~_target).sum() / nps
    false_negative = (~_pred & _target).sum() / nps

    # Compute recall, precision and quality
    accuracy = 100. * (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    quality = 100. * true_positive / (true_positive + false_positive + false_negative)
    recall = 100. * true_positive / (true_positive + false_negative)
    
    return accuracy, quality, recall

