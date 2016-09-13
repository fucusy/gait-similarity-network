import sys
import config
from preprocess.resize import resize_image_main
from preprocess.argument import argument_main
from preprocess.resize import add_padding_main
from preprocess.resize import crop_image_main
from tool.data_tools import compute_mean_image

from tool.model_tools import KerasModel
from tool.keras_tool import load_data
import model.cnn_model as model_factory


import logging

if __name__ == '__main__':
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    '''=====================================Data resize=================================================='''

    # argument_main()

    # resize begin
    path_list = ["/home/chenqiang/kaggle_driver_data/imgs/train_112_112_padding_224_224"]
    save_to = "%s/cache/mean_train_112_112_padding_224_224.jpg" % config.Project.project_path
    compute_mean_image(path_list, [], save_to)
