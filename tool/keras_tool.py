__author__ = 'fucus'

import sys
sys.path.append('../')

import random
import config
import os
import numpy as np
from scipy.misc import imread, imresize
import logging
from keras.utils.np_utils import to_categorical
logger = logging.getLogger('keras_tool')


def extract_info_from_path(path):
    img_id = ''.join(os.path.basename(path).split('.')[:-1])
    split_img_id = img_id.split('-')
    hid = split_img_id[0]
    cond = split_img_id[1]
    seq = split_img_id[2]
    view = split_img_id[3]
    return hid, cond, seq, view



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
        label_list += [sub_folder] * len(result)
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
    img /= 255
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
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
    train_dat, val_dat, test_dat=load_train_validation_data_set(train_dirs)
    return train_dat, val_dat, test_dat



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

    condition = ["nm", "cl", "bg"]

    for i in range(len(img_list)):
        hid, cond, seq, view = extract_info_from_path(img_list[i])

        if cond not in condition:
            continue
        if hid in val_ids:
            validation_img_list.append(img_list[i])
            validation_img_label.append(img_label[i])
        elif hid in train_ids:
            train_img_list.append(img_list[i])
            train_img_label.append(img_label[i])
        elif hid in test_ids:
            test_img_list.append(img_list[i])
            test_img_label.append(img_label[i])

    train_data = SimiDataSet(train_img_list, train_img_label)
    val_data = SimiDataSet(validation_img_list, validation_img_label)
    test_data = SimiDataSet(test_img_list, test_img_label)
    return train_data, val_data, test_data

class SimiDataSet(object):
    def __init__(self, img_paths, img_labels, neg_pick='random'):
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
        self.labels = []
        self.label_to_imgs = {}
        self.pre_func = img_preprocess
        labels = set()
        for i in range(len(img_labels)):
            label = img_labels[i]
            img = img_paths[i]
            labels.add(label)
            if label not in self.label_to_imgs:
                self.label_to_imgs[label] = [img, ]
            else:
                self.label_to_imgs[label].append(img)

        self.labels = sorted(labels)
        for l in self.labels:
            count_l = len(self.label_to_imgs[l])
            for i in range(count_l):
                hid, cond, seq, view = extract_info_from_path(self.label_to_imgs[l][i])
                if cond != 'nm':
                    continue
                for j in range(i, count_l):
                    hid, cond, seq, view = extract_info_from_path(self.label_to_imgs[l][j])
                    if cond != 'nm':
                        continue

                    self.pairs[0].append(self.label_to_imgs[l][i])
                    self.pairs[1].append(self.label_to_imgs[l][j])
                    self.simi.append(1)

                    self.pairs[0].append(self.label_to_imgs[l][j])
                    self.pairs[1].append(self.label_to_imgs[l][i])
                    self.simi.append(1)

        random_seed = 2016
        random.seed(random_seed)
        for i in range(len(self.labels)):
            l_i = self.labels[i]
            count_i = len(self.label_to_imgs[l_i])
            for index_i in range(count_i):
                hid, cond, seq, view = extract_info_from_path(self.label_to_imgs[l_i][index_i])
                if cond != 'nm':
                    continue
                for tmp in range(count_i):
                    # randomly pick dis-similar image
                    random_j = random.randint(0, len(self.labels)-2)
                    if random_j >= i:
                        random_j += 1
                    j = random_j
                    l_j = self.labels[j]
                    count_j = len(self.label_to_imgs[l_j])
                    index_j = random.randint(0, count_j - 1)
                    hid, cond, seq, view = extract_info_from_path(self.label_to_imgs[l_j][index_j])
                    if cond != 'nm':
                        continue
                    self.pairs[0].append(self.label_to_imgs[l_i][index_i])
                    self.pairs[1].append(self.label_to_imgs[l_j][index_j])
                    self.simi.append(0)

        count_0 = 0
        count_1 = 0
        for s in self.simi:
            if s == 0:
                count_0 += 1
            elif s == 1:
                count_1 += 1
        logging.info("non similar pair count:%d, similar pair count:%d" % (count_0, count_1))
        random_seed = 2016
        np.random.seed(random_seed)
        permut = np.random.permutation(len(self.simi))
        self.pairs[0] = np.array(self.pairs[0])[permut]
        self.pairs[1] = np.array(self.pairs[1])[permut]

        self.simi = np.array(self.simi)[permut]
        logger.debug("before %s" % str(self.simi.shape))
        self.simi = self.simi.reshape((-1,1))
        #self.simi = to_categorical(self.simi, 2)
        logger.debug("before %s" % str(self.simi.shape))
    def get_probes_img_paths(self, hid, probe_view, probe_cond='nm'):
        if hid not in self.label_to_imgs.keys():
            return []
        imgs = []
        for img_path in self.label_to_imgs[hid]:
            hid, cond, seq, view = extract_info_from_path(img_path)
            if probe_cond == 'nm':
                probe_seq = ['05', '06']
            else:
                probe_seq = ['01', '02']
            if view == probe_view and cond == probe_cond and seq in probe_seq:
                imgs.append(img_path)
        return imgs

    def get_probes(self, hid, view, cond='nm'):
        """
        get imgs of nm-05 nm-06 of this hid and this probe_view
        hid: string, the human id, such as '001'
        probe_view: string, the view, such as '018'
        return a numpy of imgs data
        """
        imgs = self.get_probes_img_paths(hid, view, cond)
        return self.img_path_2_pic(imgs)

    def get_cloth_probes(self, hid, view):
        return self.get_probes(hid, view, cond='cl')

    def get_bag_probes(self, hid, view):
        return self.get_probes(hid, view, cond='bg')

    def get_gallery_paths(self, g_view=None):
        paths = []
        labels = []
        for i in range(len(self.img_paths)):
            img_p = self.img_paths[i]
            img_l = self.img_labels[i]
            hid, cond, seq, view = extract_info_from_path(img_p)
            if cond == 'nm' and seq in ['01','02','03','04']:
                if g_view is None or view == g_view:
                    paths.append(img_p)
                    labels.append(img_l)
        return paths, labels

    def get_gallerys(self, view=None):
        """
        get imgs of nm-01 nm-02 nm-03 nm-04 of this probe_view
        view: string, the view, such as '018'
        return numpy of list imgs and numpy of list of hids
        """
        paths, labels = self.get_gallery_paths(view)
        datas = self.img_path_2_pic(paths)
        return datas, np.array(labels) 

    def get_simi(self):
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

    def next_fragment(self, fragment_size, need_label=True):
        pairs = [[], []]
        start = self.index_in_epoch
        end = min(self.index_in_epoch + fragment_size, self.count())
        self.index_in_epoch = end
        pairs[0] = self.pairs[0][start:end]
        pairs[1] = self.pairs[1][start:end]

        feature_list = self.pairs_2_pic(pairs)

        img_paths = [[], []]
        for i in range(2):
            img_paths[i] = [os.path.basename(x) for x in pairs[i]]

        if need_label and self.simi is not None:
            target = self.simi[start:end]
            logger.debug("target, %s" % str(target.shape))
            return feature_list, target, img_paths
        else:
            return feature_list, img_paths

    def img_path_2_pic(self, img_path_list):
        img_pic_list = []
        for img_path in img_path_list:
            im = imread(img_path)
            if self.pre_func is not None:
                im = self.pre_func(im)
            img_pic_list.append(im)
        return np.array(img_pic_list)

    def pairs_2_pic(self, pairs):
        return [self.img_path_2_pic(pairs[0]), self.img_path_2_pic(pairs[1])]

if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    train_img_dirs = config.data.train_img_dirs 
    print train_img_dirs 
    train_data, validation_data, _ = load_data(train_img_dirs)
    while train_data.have_next():
        img_list, img_label, _ = train_data.next_fragment(2,need_label=True)
        for n in img_list[0][0]:
            for m in n:
                print m
            break
        print(img_list[0][0])
        print(img_label)
        view = "018"
        label = train_data.labels[0]
        imgs = train_data.get_probes(label, view, 'nm')
        print("probe nm imgs of view:%s label:%s" % (view, label))
        print(imgs)

        imgs = train_data.get_probes(label, view, 'cl')
        print("probe cl imgs of view:%s label:%s" % (view, label))
        print(imgs)

        break
