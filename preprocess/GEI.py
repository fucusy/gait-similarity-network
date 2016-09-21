__author__ = 'fucus'

import skimage.io
from skimage.io import imread
from skimage.io import imsave
from scipy.misc import imresize
import numpy as np
import os
import logging

logger = logging.getLogger("tool")


def shift_left(img, left=10.0, is_grey=True):
    """

    :param numpy.array img: represented by numpy.array
    :param float left: how many pixels to shift to left, this value can be negative that means shift to
                    right {-left} pixels
    :return: numpy.array
    """
    if 0 < abs(left) < 1:
        left = int(left * img.shape[1])
    else:
        left = int(left)

    img_shift_left = np.zeros(img.shape)
    if left >= 0:
        if is_grey:
            img_shift_left = img[:, left:]
        else:
            img_shift_left = img[:, left:, :]
    else:
        if is_grey:
            img_shift_left = img[:, :left]
        else:
            img_shift_left = img[:, :left, :]

    return img_shift_left


def shift_right(img, right=10.0):
    return shift_left(img, -right)


def shift_up(img, up=10.0, is_grey=True):
    """
    :param numpy.array img: represented by numpy.array
    :param float up: how many pixels to shift to up, this value can be negative that means shift to
                    down {-up} pixels
    :return: numpy.array
    """


    if 0 < abs(up) < 1:
        up = int(up * img.shape[0])
    else:
        up = int(up)

    img_shift_up = np.zeros(img.shape)
    if up >= 0:
        if is_grey:
            img_shift_up = img[up:, :]
        else:
            img_shift_up = img[up:, :, :]
    else:
        if is_grey:
            img_shift_up = img[:up, :]
        else:
            img_shift_up = img[:up, :, :]

    return img_shift_up

def shift_down(img, down=10.0):
    return shift_up(img, -down)


def load_image_path_list(path):
    """
    :param path: the test image folder
    :return:
    """
    list_path = os.listdir(path)
    result = ["%s/%s" % (path, x) for x in list_path if x.endswith("jpg") or x.endswith("png")]
    return result



def image_path_list_to_image_pic_list(image_path_list):
    image_pic_list = []
    for image_path in image_path_list:
        im = imread(image_path)
        image_pic_list.append(im)
    return image_pic_list

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

    img = shift_left(img, left_blank)
    img = shift_right(img, right_blank)
    img = shift_up(img, up_blank)
    img = shift_down(img, down_blank)
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

def build_GEI(img_list):
    """
    :param img_list: a list of grey image numpy.array data
    :return:
    """
    norm_width = 70
    norm_height = 210
    result = np.zeros((norm_height, norm_width), dtype=np.int)

    human_extract_list = []
    for img in img_list:
        try:
            human_extract_list.append(center_person(extract_human(img), (norm_height, norm_width)))
        except:
            logger.warning("fail to extract human from image")
    try:
        result = np.mean(human_extract_list, axis=0)
    except:
        logger.warning("fail to calculate GEI, return an empty image")

    return result.astype(np.int)

def img_path_to_GEI(img_path):
    """
    convert the images in the img_path to GEI
    :param img_path: string
    :return: a GEI image
    """

    id = img_path.replace("/", "_")
    img_list = load_image_path_list(img_path)
    img_data_list = image_path_list_to_image_pic_list(img_list)
    GEI_image = build_GEI(img_data_list)
    return GEI_image

if __name__ == '__main__':
    origin_dir = "/home/chenqiang/data/CASIA_full_gait_data/DatasetB/silhouettes"
    GEI_dir = "/home/chenqiang/data/CASIA_gait_data_GEI"
    for hid in ["%03d" % x for x in range(1, 125)]:
        for style in ["bg", "cl", "nm"]:
            if style == "nm":
                seq_count = 6
            else:
                seq_count = 2
            for seq in ["%02d" % x for x in range(1, seq_count+1)]:
                for angle in ["%03d" % x for x in range(0, 181, 18)]:
                    final_dir = "%s/%s/%s-%s/%s" %\
                             (origin_dir, hid, style, seq, angle)
                    print(final_dir)
                    try:
                        GEI_image = img_path_to_GEI(final_dir)
                    except Exception,e:
                        print(str(e))
                        print("error to generate GEI")
                    try:
                        target_file = "%s/%s/%s-%s-%s-%s.bmp" %\
                                (GEI_dir, hid, hid, style, seq, angle)
                        target_path = os.path.dirname(target_file)
                        if not os.path.exists(target_path):
                            os.makedirs(target_path)
                        imsave(target_file, GEI_image)
                    except Exception,e:
                        print(str(e))
                        print("error to save GEI")