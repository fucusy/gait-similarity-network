__author__ = 'fucus'

import sys
sys.path.append('../')

import config
from keras.models import model_from_yaml
import os
import numpy as np
from scipy.misc import imread, imresize
import logging
from keras.utils.np_utils import to_categorical



def save_model(model, weight_path, structure_path=''):
    """
    save model to file system
    :param model, the model
    :param weight_path, the weight path file you want, required
    :param structure_path, the structure  path file you want, optional
    """
    model_string = model.to_yaml()
    if structure_path == '':
        structure_path = weight_path + ".yaml"
    open(structure_path, 'w').write(model_string)
    model.save_weights(weight_path, overwrite=True)

def load_model(weight_path, structure_path=''):
    """
    load the keras model, from your saved model

    :return: uncompile model
    """
    if structure_path == '':
        structure_path = weight_path + ".yaml"
    model = model_from_yaml(open(structure_path).read())
    model.load_weights(weight_path)
    return model

def load_img_path_list(path):
    """

    :param path: the test img folder
    :return:
    """
    list_path = os.listdir(path)
    result = ["%s/%s" % (path, x) for x in list_path if x.endswith("jpg") or x.endswith("png") or x.endswith("bmp")]
    return np.array(result)


def load_train_img_path_list_and_label(train_path):
    label_list = []
    result_list = []
    for x in range(1, 125):
        sub_folder = '%03d' % x
        path = "%s/%s" % (train_path, sub_folder)
        result = load_img_path_list(path)
        label_list += [x-1] * len(result)
        result_list += list(result)
    return np.array(result_list), np.array(label_list)

####  preprocess function

def resize_and_mean(img, size=(224, 224), mean=(103.939, 116.779, 123.68)):
    """
    :param img:
    :param size:
    :param mean:
    :return:
    """
    img_resized = imresize(img, size)
    img_resized = img_resized.transpose((2, 0, 1))

    for c in range(3):
        img_resized[c, :, :] = img_resized[c, :, :] - mean[c]
    return img_resized

def normalization_grey_image(image):
    image = image.astype(np.float32)
    image /= 255
    return image


def img_preprocess(img):
    """

    :param img:
    :param mean: the mean img computed by  data_tool.compute_mean_img
    :return:
    """
    img = img.astype(np.float32)
    mean = imread(config.Data.mean_img_file_name)
    img -= mean
    img = img.transpose((2, 0, 1))

    return img

def load_test_data_set(test_img_path):
    test_img_list = load_img_path_list(test_img_path)
    return DataSet(test_img_list)


def load_data(train_dirs):
    """

    :param train_dirs:
    :param test_dirs:
    :return: three DataSet structure include train data, validation data, test data
    """
    train_data, validation_data = load_train_validation_data_set(train_dirs)

    return train_data, validation_data



def load_train_validation_data_set(path):
    """
    001-050 as train data
    051-074 as evaluate data
    075-124 as as test data
    return a tuple of dataset contain train data set and validation data set
    """

    if type(path) is list:
        logging.debug("train validation data from multi-directory %s" % ",".join(path))
        img_list = []
        img_label = []
        for p in path:
            img_list_tmp, img_label_tmp = load_train_img_path_list_and_label(p)
            img_list += list(img_list_tmp)
            img_label += list(img_label_tmp)

    else:
        logging.debug("train validation data from %s" % path)
        img_list, img_label = load_train_img_path_list_and_label(path)

    train_img_list = []
    train_img_label = []

    validation_img_list = []
    validation_img_label = []

    test_img_list = []
    test_img_label = []

    train_ids = ["%03d" % x for x in range(1, 51)]
    val_ids = ["%03d" % x for x in range(51, 75)]
    test_ids = ["%03d" % x for x in range(75, 124)]

    for i in range(len(img_list)):
        img_id = os.path.basename(img_list[i]).split('.')[0]
        id = img_id.split("-")[0]
        if id in val_ids:
            validation_img_list.append(img_list[i])
            validation_img_label.append(img_label[i])
        elif id in train_ids:
            train_img_list.append(img_list[i])
            train_img_label.append(img_label[i])
        elif id in test_ids:
            test_img_list.append(img_list[i])
            test_img_label.append(img_label[i])

    train_data = SimiDataSet(train_img_list, train_img_label)
    val_data = SimiDataSet(validation_img_list, validation_img_label)
    # test_data = DataSet(test_img_list, test_img_label)
    return train_data, val_data

class SimiDataSet(object):
    def __init__(self, img_paths, img_labels):
        """

        :param img_paths:
        :param img_labels:
        :return:
        """
        self.epoch_completed = False
        self.index_in_epoch = 0
        self.pairs = [[],[]]
        self.simi = []
        self.img_paths = img_paths
        self.img_labels = img_labels
        labels = set()
        label_to_imgs = {}
        for i in range(len(img_labels)):
            label = img_labels[i]
            img = img_paths[i]
            labels.add(label)
            if label not in label_to_imgs:
                label_to_imgs[label] = [img, ]
            else:
                label_to_imgs[label].append(img)

        labels = sorted(labels)
        for l in labels:
            count_l = len(label_to_imgs[l])
            for i in range(count_l):
                for j in range(i, count_l):
                    self.pairs[0].append(label_to_imgs[l][i])
                    self.pairs[1].append(label_to_imgs[l][j])
                    self.simi.append(1)

                    self.pairs[0].append(label_to_imgs[l][j])
                    self.pairs[1].append(label_to_imgs[l][i])
                    self.simi.append(1)

        for i in range(len(labels)):
            j = (i+1) % len(labels)
            l_i = labels[i]
            l_j = labels[j]
            count_i = len(label_to_imgs[l_i])
            count_j = len(label_to_imgs[l_j])
            for index_i in range(count_i):
                for index_j in range(count_j):
                    self.pairs[0].append(label_to_imgs[l_i][index_i])
                    self.pairs[1].append(label_to_imgs[l_j][index_j])
                    self.simi.append(0)

        random = 2016
        np.random.seed(random)
        permut = np.random.permutation(len(self.simi))
        self.pairs[0] = np.array(self.pairs[0])[permut]
        self.pairs[1] = np.array(self.pairs[1])[permut]

        self.simi = np.array(self.simi)[permut]
        print("before", self.simi, self.simi.shape)
        self.simi = to_categorical(self.simi, 2)
        print("after", self.simi, self.simi.shape)

    def get_labels(self):
        return self.simi

    def count(self):
        return len(self.simi)

    def set_index_in_epoch(self, index):
        self.index_in_epoch = index

    def reset_index(self):
        self.set_index_in_epoch(0)

    def have_next(self):
        return self.index_in_epoch < self.count()

    def load_all_img(self, need_label=False):
        index_in_epoch = self.index_in_epoch
        self.reset_index()
        result = self.next_fragment(self.count(), need_label)
        self.set_index_in_epoch(index_in_epoch)
        return result

    def next_fragment(self, fragment_size, preprocess_fuc=None, need_label=True):
        pairs = [[], []]
        start = self.index_in_epoch
        end = min(self.index_in_epoch + fragment_size, self.count())
        self.index_in_epoch = end
        pairs[0] = self.pairs[0][start:end]
        pairs[1] = self.pairs[1][start:end]

        feature_list = self.pairs_2_pic(pairs, preprocess_fuc)
        for i in range(2):
            if len(feature_list[i].shape) == 3:
                feature_list[i] = feature_list[i].reshape(feature_list[i].shape[0], feature_list[i].shape[1], feature_list[i].shape[2], 1)

            print "feature_list shape"
            print feature_list[i].shape
        img_paths = [[], []]
        for i in range(2):
            img_paths[i] = [os.path.basename(x) for x in pairs[i]]

        if need_label and self.simi is not None:
            target = self.simi[start:end]
            print("target, ", target, target.shape)
            return feature_list, target, img_paths
        else:
            return feature_list, img_paths

    def img_path_2_pic(self, img_path_list, func=None):
        img_pic_list = []
        for img_path in img_path_list:
            im = imread(img_path)
            if func is not None:
                im = func(im)
            img_pic_list.append(im)
        return np.array(img_pic_list)

    def pairs_2_pic(self, pairs, func=None):
        return [self.img_path_2_pic(pairs[0], func), self.img_path_2_pic(pairs[1], func)]

if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    train_img_dirs = config.data.train_img_dirs 
    print train_img_dirs 
    train_data, validation_data = load_data(train_img_dirs)
    while train_data.have_next():
        img_list, img_label, _ = train_data.next_fragment(2, need_label=True,preprocess_fuc=normalization_grey_image)
        for n in img_list[0][0]:
            for m in n:
                print m
        print(img_list[0][0])
        print(img_label)
        break
