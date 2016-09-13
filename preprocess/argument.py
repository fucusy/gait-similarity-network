__author__ = 'fucus'


import sys
sys.path.append("../")

import numpy as np
from skimage.io import imsave
from skimage.io import imread
import config
import os
import shutil
import skimage.io as skio
import logging
from tool.keras_tool import load_image_path_list
def shift_left(img, left=10.0, is_grey=False):
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


def shift_right(img, right=10.0, is_grey=False):
    return shift_left(img, -right, is_grey)


def shift_up(img, up=10.0, is_grey=False):
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

def shift_down(img, down=10.0, is_grey=False):
    return shift_up(img, -down, is_grey)

def loop_process_test_image(from_path, to_path, method, args, force=False):
    if force:
        if os.path.exists(to_path):
            shutil.rmtree(to_path)
    if os.path.exists(to_path):
        logging.info("save path exists, no need to process again, skip this")
        return
    else:
        os.makedirs(to_path)

    logging.info("doing process now, saving  result to %s" % os.path.basename(to_path))

    total = 0
    file_paths = load_image_path_list(from_path)
    total += len(file_paths)

    count = 0
    file_paths = load_image_path_list(from_path)
    for file_path in file_paths:
        count += 1
        if count % 1000 == 0:
            logging.debug("process %d/%d %.3f%% image" % (count, total, count * 100.0 / total))
        f = os.path.basename(file_path)
        img = skio.imread(file_path)
        args["img"] = img
        img = method(**args)
        save_path = os.path.join(to_path, f)
        skio.imsave(save_path, img)
    logging.info("process done saving data to %s" % os.path.basename(to_path))

def loop_process_train_image(from_path, to_path, method, args, force=False):
    if force:
        if os.path.exists(to_path):
            shutil.rmtree(to_path)
    if os.path.exists(to_path):
        logging.info("save path exists, no need to process again, skip this")
        return

    class_list = ["%03d" % x for x in range(1, 125)]
    logging.info("doing process now, saving  result to %s" % os.path.basename(to_path))
    if not os.path.exists(to_path):
        for c in class_list:
            class_path = os.path.join(to_path, c)
            os.makedirs(class_path)
    total = 0
    for c in class_list:
        class_path = os.path.join(from_path, c)
        file_paths = load_image_path_list(class_path)
        total += len(file_paths)

    count = 0
    for c in class_list:
        class_path = os.path.join(from_path, c)
        file_paths = load_image_path_list(class_path)
        for file_path in file_paths:
            count += 1
            if count % 1000 == 0:
                logging.debug("process %d/%d %.3f%% image" % (count, total, count * 100.0 / total))
            f = os.path.basename(file_path)
            img = skio.imread(file_path)
            args["img"] = img
            try:
                img = method(**args)
                save_path = os.path.join(to_path, c, f)
                skio.imsave(save_path, img)
            except:
                logging.warning("fail to process %s" % file_path)

    logging.info("process done saving data to %s" % os.path.basename(to_path))


def argument_main():
    from_path = config.Project.original_training_folder
    to_path = "%s_shift_left_0.2" % from_path
    loop_process_train_image(from_path, to_path, shift_left, {"left": 0.2})

    to_path = "%s_shift_right_0.2" % from_path
    loop_process_train_image(from_path, to_path, shift_right, {"right": 0.2})

    to_path = "%s_shift_up_0.2" % from_path
    loop_process_train_image(from_path, to_path, shift_up, {"up": 0.2})

    to_path = "%s_shift_down_0.2" % from_path
    loop_process_train_image(from_path, to_path, shift_down, {"down": 0.2})


if __name__ == '__main__':
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    driver = imread(config.Project.test_img_example_path)    
    driver_shift_left = shift_left(driver, 0.2)
    driver_shift_right = shift_right(driver, 0.2)

    driver_shift_up = shift_up(driver, 0.2)
    driver_shift_down = shift_down(driver, 0.2)

    imsave("driver.jpg", driver)
    imsave("driver_shift_left.jpg", driver_shift_left)
    imsave("driver_shift_right.jpg", driver_shift_right)
    imsave("driver_shift_up.jpg", driver_shift_up)
    imsave("driver_shift_down.jpg", driver_shift_down)
    argument_main()
