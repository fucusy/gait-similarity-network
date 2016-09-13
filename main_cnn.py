#!/usr/bin/python2.7

import sys
import config

from tool.model_tools import KerasModel
from tool.keras_tool import load_data
import models as model_factory
from tool.keras_tool import normalization_grey_image


import logging

if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    train_img_dirs = ["/Volumes/Passport/data/gait-simple-cnn-data_extract_210_70"]

    train_data, validation_data = load_data(train_img_dirs)

    logging.info("train data image count %s" % train_data.count())
    logging.info("validation data image count %s" % validation_data.count())

    model_func = getattr(model_factory, config.CNN.model_name)

    weights_path = config.CNN.keras_train_weight
    lr = config.CNN.lr
    cnn_model = model_func(lr, weights_path=None)

    model = KerasModel(cnn_model=cnn_model, preprocess_func=normalization_grey_image)
    model.train_model(train_data, validation_data, save_best=True)
