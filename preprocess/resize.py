#!/usr/bin/python2.7

__author__ = 'fucus'

import sys
sys.path.append("../")

import config
import os
import logging
import skimage.io as skio
import skimage.transform as sktr
import shutil
from tool.keras_tool import load_image_path_list
from preprocess.argument import loop_process_train_image
from preprocess.argument import loop_process_test_image
from preprocess.argument import shift_left
from preprocess.argument import shift_down
from preprocess.argument import shift_right
from preprocess.argument import shift_up
import numpy as np
from scipy.misc import imresize


'''
    Save all resized images to training and testing folders.
    training images with all labels are saved into one folder, the label is the first character of the file names.
'''

def resize_image(img, img_size):
    return sktr.resize(img, output_shape=img_size)


def add_padding(img, img_size):
    result = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    origin_size = img.shape
    offset_height = (img_size[0] - origin_size[0]) / 2
    offset_width = (img_size[1] - origin_size[1]) / 2

    for w in range(origin_size[0]):
        for h in range(origin_size[1]):
            result[w + offset_height, h + offset_width, :] = img[w, h, :]

    return result

def crop_center(img, img_size):

    height, width = img.shape[0], img.shape[1]

    height_offset = (height - img_size[0]) / 2
    width_offset = (width - img_size[1]) / 2

    corp_img = img[height_offset:height_offset + img_size[0], width_offset:width_offset + img_size[1], :]
    return corp_img

def extract_human(img):
    """

    :param img: grey type numpy.array image
    :return:
    """

    left_blank = 0
    right_blank = 0

    up_blank = 0
    down_blank = 0

    height = img.shape[0]
    width = img.shape[1]

    for i in range(height):
        if np.sum(img[i, :]) == 0:
            up_blank += 1
        else:
            break

    for i in range(height-1, -1, -1):
        if np.sum(img[i, :]) == 0:
            down_blank += 1
        else:
            break

    for i in range(width):
        if np.sum(img[:, i]) == 0:
            left_blank += 1
        else:
            break

    for i in range(width-1, -1, -1):
        if np.sum(img[:, i]) == 0:
            right_blank += 1
        else:
            break

    is_grey = True
    img = shift_left(img, left_blank, is_grey)
    img = shift_right(img, right_blank, is_grey)
    img = shift_up(img, up_blank, is_grey)
    img = shift_down(img, down_blank, is_grey)
    return img

def center_person(img, size):
    """

    :param img: grey image, numpy.array datatype
    :param size: tuple, for example(120, 160), first number for height, second for width
    :return:
    """

    highest_index = 0
    highest = 0
    origin_height, origin_width = img.shape

    for i in range(origin_width):
        data = img[:, i]
        for j, val in enumerate(data):

            # encounter body
            if val > 0:
                now_height = origin_height - j
                if now_height > highest:
                    highest = now_height
                    highest_index = i
                break

    left_part_column_count = highest_index
    right_part_column_count = origin_width - left_part_column_count - 1

    if left_part_column_count == right_part_column_count:
        return imresize(img, size)
    elif left_part_column_count > right_part_column_count:
        right_padding_column_count = left_part_column_count - right_part_column_count
        new_img = np.zeros((origin_height, origin_width + right_padding_column_count), dtype=np.int)
        new_img[:, :origin_width] = img
    else:
        left_padding_column_count = right_part_column_count - left_part_column_count
        new_img = np.zeros((origin_height, origin_width + left_padding_column_count), dtype=np.int)
        new_img[:, left_padding_column_count:] = img

    return imresize(new_img, size)

def add_padding_main(from_path, img_size, force=False):
    for path in from_path:
        img_size_str_list = [str(x) for x in img_size]
        to_path = "%s_padding_%s" % (path, "_".join(img_size_str_list))
        args = {"img_size": img_size}
        logging.info("add padding image in %s to %s" % (path, to_path))
        if 'c0' in os.listdir(path):
            loop_process_train_image(path, to_path, add_padding, args, force)
        else:
            loop_process_test_image(path, to_path, add_padding, args, force)


def resize_image_main(from_path, img_size, force=False):
    for path in from_path:
        img_size_str_list = [str(x) for x in img_size]
        to_path = "%s_%s" % (path, "_".join(img_size_str_list))
        args = {"img_size": img_size}
        logging.info("resize image in %s to %s" % (path, to_path))
        if 'c0' in os.listdir(path):
            loop_process_train_image(path, to_path, resize_image, args, force)
        else:
            loop_process_test_image(path, to_path, resize_image, args, force)



def crop_image_main(from_path, img_size, force=False):
    for path in from_path:
        img_size_str_list = [str(x) for x in img_size]
        to_path = "%s_crop_%s" % (path, "_".join(img_size_str_list))
        args = {"img_size": img_size}
        logging.info("crop image in %s to %s" % (path, to_path))
        if 'c0' in os.listdir(path):
            loop_process_train_image(path, to_path, crop_center, args, force)
        else:
            loop_process_test_image(path, to_path, crop_center, args, force)

def extract_human_center(img, img_size):
    return center_person(extract_human(img), img_size)

def extract_human_center_main(from_path, img_size, force=False):
    for path in from_path:
        img_size_str_list = [str(x) for x in img_size]
        to_path = "%s_extract_%s" % (path, "_".join(img_size_str_list))
        args = {"img_size": img_size}
        if '001' in os.listdir(path):
            loop_process_train_image(path, to_path, extract_human_center, args, force)
        else:
            loop_process_test_image(path, to_path, extract_human_center, args, force)

     

if __name__ == "__main__":
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    img_size = (210, 70)
    from_path = ["/Volumes/Passport/data/gait-simple-cnn-data"]
    force = True
    extract_human_center_main(from_path, img_size, force)
