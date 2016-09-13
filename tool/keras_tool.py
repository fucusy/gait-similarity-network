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

def load_image_path_list(path):
    """

    :param path: the test image folder
    :return:
    """
    list_path = os.listdir(path)
    result = ["%s/%s" % (path, x) for x in list_path if x.endswith("jpg") or x.endswith("png")]
    return np.array(result)


def load_train_image_path_list_and_label(train_path):
    label_list = []
    result_list = []
    for x in range(1, 125):
        sub_folder = '%03d' % x
        path = "%s/%s" % (train_path, sub_folder)
        result = load_image_path_list(path)
        label_list += [x-1] * len(result)
        result_list += list(result)
    return np.array(result_list), np.array(label_list)

####  preprocess function

def resize_and_mean(image, size=(224, 224), mean=(103.939, 116.779, 123.68)):
    """
    :param image:
    :param size:
    :param mean:
    :return:
    """
    img_resized = imresize(image, size)
    img_resized = img_resized.transpose((2, 0, 1))

    for c in range(3):
        img_resized[c, :, :] = img_resized[c, :, :] - mean[c]
    return img_resized

def normalization_grey_image(image):
    image = image.astype(np.float32)
    image /= 255
    return image


def image_preprocess(image):
    """

    :param image:
    :param mean: the mean img computed by  data_tool.compute_mean_image
    :return:
    """
    image = image.astype(np.float32)
    mean = imread(config.Data.mean_image_file_name)
    image -= mean
    image = image.transpose((2, 0, 1))

    return image

def load_test_data_set(test_image_path):
    test_image_list = load_image_path_list(test_image_path)
    return DataSet(test_image_list)


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
        image_list = []
        image_label = []
        for p in path:
            image_list_tmp, image_label_tmp = load_train_image_path_list_and_label(p)
            image_list += list(image_list_tmp)
            image_label += list(image_label_tmp)

    else:
        logging.debug("train validation data from %s" % path)
        image_list, image_label = load_train_image_path_list_and_label(path)

    train_image_list = []
    train_image_label = []

    validation_image_list = []
    validation_image_label = []

    test_image_list = []
    test_image_label = []

    train_ids = ["%03d" % x for x in range(1, 51)]
    val_ids = ["%03d" % x for x in range(51, 75)]
    test_ids = ["%03d" % x for x in range(75, 124)]

    for i in range(len(image_list)):
        image_id = os.path.basename(image_list[i]).split('.')[0]
        id = image_id.split("-")[0]
        if id in val_ids:
            validation_image_list.append(image_list[i])
            validation_image_label.append(image_label[i])
        elif id in train_ids:
            train_image_list.append(image_list[i])
            train_image_label.append(image_label[i])
        elif id in test_ids:
            test_image_list.append(img_list[i])
            test_image_label.append(image_label[i])

    train_data = SimiDataSet(train_image_list, train_image_label)
    val_data = SimiDataSet(validation_image_list, validation_image_label)
    # test_data = DataSet(test_image_list, test_image_label)
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
                    self.simi.append([0, 1])
        for i in range(len(labels)):
            j = (i+1) % len(labels)
            count_i = len(label_to_imgs[i])
            count_j = len(label_to_imgs[j])
            for index_i in range(count_i):
                for index_j in range(count_j):
                    self.pairs[0].append(label_to_imgs[i][index_i])
                    self.pairs[1].append(label_to_imgs[j][index_j])
                    self.simi.append([1, 0])

        random = 2016
        np.random.seed(random)
        permut = np.random.permutation(len(self.pairs))
        self.pairs[0] = self.pairs[0][permut]
        self.pairs[1] = self.pairs[1][permut]
        self.simi = self.simi[permut]

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

    def load_all_image(self, need_label=False):
        index_in_epoch = self.index_in_epoch
        self.reset_index()
        result = self.next_fragment(self.count(), need_label)
        self.set_index_in_epoch(index_in_epoch)
        return result

    def next_fragment(self, fragment_size, preprocess_fuc=None, need_label=False):
        pairs = [[], []]
        start = self.index_in_epoch
        end = min(self.index_in_epoch + fragment_size, self.count())
        self.index_in_epoch = end
        pairs[0] = self.pairs[0][start:end]
        pairs[1] = self.pairs[1][start:end]

        feature_list = self.pairs_2_pic(pairs, preprocess_fuc)

        for i in range(2):
            if len(feature_list[i].shape) == 3:
                feature_list[i] = feature_list[i].reshape(feature_list[i].shape[0], 1, feature_list[i].shape[1], feature_list[i].shape[2])

        img_paths = [[], []]
        for i in range(2):
            img_paths[i] = [os.path.basename(x) for x in pairs[i]]

        if need_label and self.img_labels is not None:
            return feature_list, self.img_labels[start:end], img_paths
        else:
            return feature_list, img_paths

    def image_path_2_pic(self, image_path_list, func=None):
        image_pic_list = []
        for image_path in image_path_list:
            im = imread(image_path)
            if func is not None:
                im = func(im)
            image_pic_list.append(im)
        return np.array(image_pic_list)

    def pairs_2_pic(self, pairs, func=None):
        return [self.image_path_2_pic(pairs[0]), self.image_path_2_pic(pairs[1])]


class DataSet(object):
    def __init__(self,
               images_path_list, image_label_list=None, to_category=False, feature_dir=None):
        """
        :param images_path_list: numpy.array or list
        :param labels: numpy.array or list
        :return:
        """
        if type(images_path_list) is list:
            images_path_list = np.array(images_path_list)

        if image_label_list is not None:
            self._one_images_label = np.array(image_label_list)
            if to_category:
                image_label_list = to_categorical(np.array(image_label_list), 124)
            else:
                image_label_list = np.array(image_label_list)

        self._num_examples = images_path_list.shape[0]
        self._images_path = images_path_list
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.feature_dir = feature_dir
        if image_label_list is not None:
            random = 2016
            np.random.seed(random)
            permut = np.random.permutation(len(images_path_list))
            self._images_path = images_path_list[permut]
            self._images_label = image_label_list[permut]
            self._one_images_label = self._one_images_label[permut]
    @property
    def image_path_list(self):
        return self._images_path

    @property
    def image_label_list(self):
        return self._images_label

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def get_image_label(self, to_cate=True):
        if to_cate:
            return self._images_label
        else:
            return self._one_images_label

    def count(self):
        return self._num_examples

    def set_index_in_epoch(self, index):
        self._index_in_epoch = index
    def reset_index(self):
        self.set_index_in_epoch(0)

    def image_path_2_pic(self, image_path_list, func=None):
        image_pic_list = []
        for image_path in image_path_list:
            im = imread(image_path)
            if func is not None:
                im = func(im)
            image_pic_list.append(im)
        return np.array(image_pic_list)

    def have_next(self):
        return self._index_in_epoch < self._num_examples
    def load_all_image(self, need_label=False):
        index_in_epoch = self._index_in_epoch
        self.reset_index()
        result = self.next_fragment(self.num_examples, need_label)
        self.set_index_in_epoch(index_in_epoch)
        return result

    def next_fragment(self, fragment_size, preprocess_fuc=None, need_label=False):

        start = self._index_in_epoch
        end = min(self._index_in_epoch + fragment_size, self._num_examples)
        self._index_in_epoch = end
        image_paths = self._images_path[start:end]
        if self.feature_dir is None:
            feature_list = self.image_path_2_pic(image_paths, preprocess_fuc)
        else:
            feature_list = self.image_path_2_feature(image_paths)

        if len(feature_list.shape) == 3:
            feature_list = feature_list.reshape(feature_list.shape[0], 1, feature_list.shape[1], feature_list.shape[2])
        image_paths = [os.path.basename(x) for x in image_paths]
        if need_label and self._images_label is not None:
            return feature_list, self._images_label[start:end], image_paths
        else:
            return feature_list, image_paths


    def image_path_2_feature(self, image_paths):
        numpy_dirs = self.feature_dir
        feature_list = []
        for image_path in image_paths:
            feature = np.array([])
            image_path = os.path.basename(image_path)
            for d in numpy_dirs:
                filename = "%s/%s.npy" % (d, image_path)
                if os.path.exists(filename) and os.path.isfile(filename):
                    feature = np.append(feature, np.load(filename), axis=0)
                else:
                    logging.error("feature file %s does not exist" % filename)
            feature_list.append(feature)

        feature_list = np.array(feature_list)
        return feature_list



if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    cache_path = "%s/cache" % config.Project.project_path
    train_dir = config.Project.train_img_folder_path
    test_dir = config.Project.test_img_folder_path
    feature_dir = [
                "vgg_feature_l_31_crop_336_336_224_224",
                "hand_feature_crop_336_336_224_224_get_hog",
                "vgg_feature_l_31_crop_336_336_112_112_padding_224_224",
                "vgg_feature_l_36_crop_336_336_224_224"
                ]
    feature_dir = ["%s/%s" % (cache_path, x) for x in feature_dir]
    train, validation, test = load_data(train_dir, test_dir, feature_dir)
    while train.have_next():
        img_list, img_label, _ = train.next_fragment(2, need_label=True)
        print(img_list)
        print(img_label)
        break
