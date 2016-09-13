# -*- coding: utf-8 -*-
"""
Created on Tue May 31 20:43:33 2016

@author: ZhengLiu
"""


import sys
sys.path.append('../')

import config
import numpy as np
import skimage
import skimage.io as skio
from skimage.io import imsave
import os
import logging

from tool.keras_tool import load_image_path_list

''' Make images, which maybe in shape of (n, h, w, ch) for multiple color images,
                                         (h, w, ch) for single color images,
                                         (n, h, w) for multiple gray images or
                                         (h, w) for single gray images,
    to data form that capible to keras models, which in the shape of (n, ch, h, w).
'''
def images_swap_axes(images, color_type=3):
    # images = [n, h, w, ch] or [h, w, ch] or [n, h, w] or [h, w]
    if color_type == 3:
        
        if len(images.shape) == 3:
            swaped_images = np.zeros((1, images.shape[2], images.shape[0], images.shape[1]))
            images = images.swapaxes(-2, -1)
            images = images.swapaxes(-3, -2)
            swaped_images[0, ...] = images
        else:
            swaped_images = np.zeros((images.shape[0], images.shape[3], images.shape[1], images.shape[2]))
            images = images.swapaxes(-2, -1)
            images = images.swapaxes(-3, -2)
            swaped_images = images

    elif color_type == 1:
        if len(images.shape) == 3:
            swaped_images = np.zeros((images.shape[0], 1, images.shape[1], images.shape[2]))
            for i in range(images.shape[0]):
                swaped_images[i, 0, ...] = images[i, ...]
        else:
            swaped_images = np.zeros((1, 1, images.shape[1], images.shape[2]))
            swaped_images[0, 0, ...] = images
    return swaped_images



'''
    Computing mean images. Using all training and testing images.
'''

def compute_mean_image(training_data_path
                       , testing_data_path
                       , save_file):

    if os.path.exists(save_file):
        logging.info("mean file already exists at %s, return it directly" % save_file)

        mean_img = skio.imread(save_file)
        logging.info("mean img is %s" % mean_img)
        return mean_img
    logging.info('computing mean images')
    folder = ["c%d" % x for x in range(10)]
    total_num = 0
    mean_image = None
    # count image first
    for train_path in training_data_path:
        for f in folder:
            folder_path = os.path.join(train_path, f)
            total_num += len(load_image_path_list(folder_path))

    for path in testing_data_path:
        total_num += len(load_image_path_list(path))

    i = 0
    for train_path in training_data_path:
        for f in folder:
            folder_path = os.path.join(train_path, f)
            for img_path in load_image_path_list(folder_path):
                i += 1
                if i % 100 == 0:
                    logging.info("process %d/%d images now" % (i, total_num))
                img = skimage.img_as_float(skio.imread(img_path))
                if mean_image is None:
                    mean_image = np.zeros(img.shape)
                mean_image += img / total_num

    for path in testing_data_path:
        for file_path in load_image_path_list(path):
            i += 1
            if i % 100 == 0:
                logging.info("process %d/%d images now" % (i, total_num))
            img = skimage.img_as_float( skio.imread(file_path))
            mean_image += img / total_num
        
    if save_file != "":
        base_path = os.path.dirname(save_file)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        with open(save_file, 'wb') as f:
            imsave(f, mean_image)
            logging.debug("saving mean file to %s" % save_file)
    print mean_image
    return mean_image

if __name__ == '__main__':

    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    '''=====================================Data resize=================================================='''


    compute_mean_image()

