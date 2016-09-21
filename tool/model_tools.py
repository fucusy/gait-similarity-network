# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:45:34 2016

@author: liuzheng
"""
import sys
sys.path.append('../')

import config
import pickle

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

import logging
import numpy as np
import csv
import skimage.io as skio
import os
from keras import backend as K
import time

logger = logging.getLogger("model_tool")

class KerasModel(object):
    def __init__(self, cnn_model, preprocess_func):
        model_name = config.CNN.model_name
        test_batch_size = config.CNN.test_batch_size
        n_iter = config.CNN.train_iter
        model_weights_file = config.CNN.model_weights_file_name
        model_save_path = config.CNN.model_save_path
        prediction_save_file = config.Project.result_output_path + "/" +time.strftime("%Y_%m_%d__%H_%M.csv")

        self._model_name = model_name
        self._n_iter = n_iter
        self._model_weights_file = model_weights_file
        self._model_save_path = model_save_path
        self._batch_size = config.CNN.batch_size
        self._test_batch_size = test_batch_size

        self.prediction_save_file = prediction_save_file
        self._prediction = {}
        self._model = cnn_model

        self.best_model_weight_path = ""
        self.min_loss = np.inf
        self.max_acc = 0.
        self.fragment_size = config.CNN.load_image_to_memory_every_time
        self.preprocess_func = preprocess_func
    
    def set_model_arch(self, model_arch):
        self._model = model_arch
    
    def set_model_weights(self, model_weights_file=''):
        self._model.load_weights(model_weights_file)
    
    def save_model_arch(self, arch_path_file):
        json_string = self._model.to_json()
        open(arch_path_file, 'w').write(json_string)
    
    def save_model_weights(self, weights_path_file, overwrite=True):
        self._model.save_weights(weights_path_file, overwrite=overwrite)
    def validate(self, validation_data):
        eva_loss = []
        eva_acc = []
        image_count = 0

        # set index to zero, prepare for have_next function
        validation_data.reset_index()
        have_print_data_shape = False
        while validation_data.have_next():
            x_valid, y_valid, _ = validation_data.next_fragment(self.fragment_size, need_label=True, preprocess_fuc=self.preprocess_func)
            if not have_print_data_shape:
                print("the input x_valid[0].shape data shape is", x_valid[0].shape)
                print("the input x_valid[0][0].shape data shape is", x_valid[0][0].shape)
                print("the input y_valid.shape data shape is", y_valid.shape)
                print("y_valid[0]:", y_valid[0])
                have_print_data_shape = not have_print_data_shape
            image_count += len(x_valid[0])
            logger.info('%s | --> validation progress %d / %d'
                  % (self._model_name, image_count, validation_data.count()))
            loss, acc = self._model.evaluate([x_valid[0], x_valid[1]], y_valid, batch_size=self._batch_size)

            eva_loss.append(loss)
            eva_acc.append(acc)

        logger.info('Compute mean evaluation loss and accuracy and output them.')
        eva_loss = float(np.mean(eva_loss))
        eva_acc = float(np.mean(eva_acc))
        logger.info('%s |  validation loss: %f, acc: %f'%(self._model_name, eva_loss, eva_acc))


        height_file_name_tpl = "%s_loss_%.3f_acc_%.3f_keras.h5"
        if eva_loss < self.min_loss:
            old_weight_path = os.path.join(self._model_save_path, height_file_name_tpl % (self._model_name, self.min_loss, self.max_acc))
            if os.path.exists(old_weight_path):
                os.remove(old_weight_path)
            self.min_loss = eva_loss
            self.max_acc = eva_acc
            new_weight_path = os.path.join(self._model_save_path, height_file_name_tpl % (self._model_name, self.min_loss, self.max_acc))
            self._model.save_weights(new_weight_path, overwrite=True)
            self.best_model_weight_path = new_weight_path

            final_weight_path = os.path.join(self._model_save_path, self._model_name + '_keras.h5')
            self._model.save_weights(final_weight_path, overwrite=True)

            logger.info('saving the best model. loss %.3f' % eva_loss)
        else:
            logger.info('the loss %.3f larger than %.3f, do not save' \
                    % (eva_loss, self.min_loss))

    def train_model(self, train_data, validation_data, save_best=True):
        json_path = os.path.join(self._model_save_path, self._model_name + '.json')
        if not os.path.exists(self._model_save_path):
            os.makedirs(self._model_save_path)

        #json_string = self._model.to_json()
        #open(json_path, 'w').write(json_string)
        pickle.dump(self._model.get_config(), open( json_path, 'w'))

        fragment_size = config.CNN.load_image_to_memory_every_time
        if fragment_size > 0:
            logger.info("can not load data into memory at once, it will load %d images every time" % fragment_size)

            logger.info("training neural network on data repeatedly %d times" % self._n_iter)
            for it in range(self._n_iter):

                logger.info("training neural network on data at %d time" % it)
                image_count = 0

                # set index to zero, prepare for have_next function
                train_data.reset_index()
                have_print_data_shape = False

                validate_every_img = config.CNN.val_every
                validate_after = validate_every_img
                while train_data.have_next():
                    x_train, y_train, _ = train_data.next_fragment(fragment_size,need_label=True, preprocess_fuc=self.preprocess_func)
                    image_count += len(x_train[0])
                    if not have_print_data_shape:
                        print("the input x_train[0].shape data shape is", x_train[0].shape)
                        print("the input x_train[0][0].shape data shape is", x_train[0][0].shape)
                        print("the input y_train.shape data shape is", y_train.shape)
                        print("y_train[0]:", y_train[0])
                        have_print_data_shape = not have_print_data_shape
                    logging.info('%s | iter %03d --> training progress  %d / %d'
                                   % (self._model_name, it, image_count, train_data.count()))
                    self._model.fit([x_train[0], x_train[1]], y_train, batch_size=self._batch_size, nb_epoch=1,  verbose=1)

                    if image_count > validate_after:
                        self.validate(validation_data)
                        validate_after += validate_every_img
                logger.info('After all training fragments are trained, evaluate the model on validation data set.')

        else:
            ''' If no need to fragment, load all training images and train the model directly.'''
            logging.info('start training')
            CallBacks = []
            if save_best:
                weight_path = os.path.join(self._model_save_path, self._model_name + '_checkpoint_weights_best_keras.h5')
                self.best_model_weight_path = weight_path
                checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
                CallBacks.append(checkpoint)

            x_train, y_train, _ = train_data.load_all_image(need_label=True)
            x_vali, y_vali, _ = validation_data.load_all_image(need_label=True)

            self._history = self._model.fit(x_train, y_train, batch_size=self._batch_size, nb_epoch=self._n_iter,
                                             validation_data=(x_vali, y_vali), callbacks=CallBacks)

    def predict_model(self, test_data):
        image_count = 0
        fragment_size = config.CNN.load_image_to_memory_every_time_when_test

        # load best model weight
        logger.info("loading best weight from %s" % self.best_model_weight_path)

        if fragment_size > 0:
            while test_data.have_next():

                x_test, name_list = test_data.next_fragment(fragment_size, need_label=False, preprocess_fuc=self.preprocess_func)
                image_count += len(x_test)
                logger.info('%s | --> testing progress %d / %d' % (self._model_name,
                                       image_count, test_data.count()))

                frag_prediction = self._model.predict(x_test, batch_size=self._test_batch_size)
                self.stat_prediction(frag_prediction, name_list)
            ''' We still call fuse function if only one prediction per image for universality.
                The self._prediction is a dict that every key (testing image names) has a list of prediction.
                No matter how many elements the list has, the final prediction is a numpy array for each image after fusing.
            '''
        else:
            x_test, name_list = test_data.load_all_images(need_label=False)
            frag_prediction = self._model.predict(x_test, batch_size=self._test_batch_size)
            self.stat_prediction(frag_prediction, name_list)
    ''' Update prediction dict. '''
    def stat_prediction(self, frag_prediction, frag_list):
        frag_prediction = np.array(frag_prediction)
        frag_list = np.array(frag_list)
        for i in range(frag_prediction.shape[0]):
            img_name = frag_list[i]
            if img_name in self._prediction.keys():
                self._prediction[img_name].append(frag_prediction[i, :])
            else:
                self._prediction[img_name] = [frag_prediction[i, :]]

    ''' In case some methods give multiple predictions to a testing image, this function fuse all predictions into one. '''
    def fuse_prediction(self):
        finalDict = {}
        for k in self._prediction.keys():
            self._prediction[k] = np.mean(np.array(self._prediction[k]), axis=0)


    ''' Save prediction dict '''
    def save_prediction(self):
        self.fuse_prediction()
        file_obj = open(self.prediction_save_file, 'wb')
        writer = csv.writer(file_obj)
        writer.writerow(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
        keys = self._prediction.keys()
        keys.sort()
        for k in keys:
            line = [k]
            for i in range(10):
                line.append('%.3f' % (self._prediction[k][i]))
            writer.writerow(line)
        file_obj.close()
        logging.info("saved result to %s" % self.prediction_save_file)
