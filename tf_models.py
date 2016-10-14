import tensorflow as tf
import logging
from tool.keras_tool import normalization_grey_image
import config
from tool.keras_tool import load_data
import os
import numpy as np
import collections

def output_res(accu_dic):
    """
    params:ordered dic, accu_dic: "nm val" => {"000":{"000":0.95, "018":0.98}, "018":{"000":0.95, "018":0.98} ...}
    the first key angle is probe_view, second key angle is gallery_view
    """
    output = ''
    for key in accu_dic.keys():
        output += '\n'
        output += '\t'.join(["type:", "gallery"] + ["%03d" % x for x in range(0, 181, 18)] + ['avg'])
        output += '\n'
        output += "%s:\tprobe" % key
        accu_sum = 0.0
        for probe_view in ["%03d" % x for x in range(0, 181, 18)]:
            key_str = "****\t%s\t" % probe_view
            accu_sum = 0.0
            for gallery_view in ["%03d" % x for x in range(0, 181, 18)]:
                if probe_view in accu_dic[key].keys()\
                    and gallery_view in accu_dic[key][probe_view]:
                    key_str += "%0.2f\t" % accu_dic[key][probe_view][gallery_view]
                    accu_sum += accu_dic[key][probe_view][gallery_view]
                else:
                    key_str += "null\t"
            key_str += '%0.2f\t' % (accu_sum / 11.0)
            output += '\n%s' % key_str
        output += '\n'
    return output
    
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def contrastive_loss(y,d):
    margin = config.CNN.margin
    part1 = y * tf.square(d)
    part2 = (1-y) * tf.square(tf.maximum((margin - d),0))
    return tf.reduce_mean(part1 + part2)

def siamses_test_share_part(x, weights, biases):
    dropout = 0.5
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=5)

    fc1 = tf.reshape(conv2, [-1, weights['out'].get_shape().as_list()[0]])
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    out = tf.nn.dropout(out, dropout)
    return out

def benchmark_part(x, weights, biases):
    conv1 = maxpool2d(x, k=2) # 105 * 35
    fc1 = tf.reshape(conv1, [-1, 105 * 35])
    return fc1

def benchmark(lr=1e-3):
    """
    return a list of var
    """
    x1 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    x2 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    y = tf.placeholder(tf.float32, [None, 1])
    # Store layers weight & bias
    output_dim = config.CNN.output_dim
    weights = {
    }
    biases = {
    }
    left = benchmark_part(x1, weights, biases)
    right = benchmark_part(x2, weights, biases)
    distance  = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(left, right),2),1,keep_dims=True))   
    loss = contrastive_loss(y, distance) 
    val_loss = contrastive_loss(y, distance) 
    optimizer = None
    return x1, x2, y, left, right, distance, loss, val_loss, optimizer

def siamses_vgg_like_part(x, weights, biases):
    dropout = 0.5
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2) # 105 * 35

    paddings = [[0, 0], [1, 0], [1, 0], [0, 0]]
    conv1 = tf.pad(conv1, paddings,mode='CONSTANT') # 106 * 36

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2) # 53 * 18

    paddings = [[0, 0], [1, 0], [0, 0], [0, 0]]
    conv2 = tf.pad(conv2, paddings,mode='CONSTANT') # 54 * 18

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)  # 27 * 9

    fc1 = tf.reshape(conv3, [-1, weights['out'].get_shape().as_list()[0]])
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def siamses_vgg_like(lr = 1e-3):
    """
    return a list of var
    """
    x1 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    x2 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    y = tf.placeholder(tf.float32, [None, 1])
    output_dim = config.CNN.output_dim
    # Store layers weight & bias
    weights = {
        # 3x3 conv, 1 input, 16 outputs
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 8])),
        'wc2': tf.Variable(tf.random_normal([3, 3, 8, 16])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 16, 32])),
        'out': tf.Variable(tf.random_normal([27*9*32, output_dim])),
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([8])),
        'bc2': tf.Variable(tf.random_normal([16])),
        'bc3': tf.Variable(tf.random_normal([32])),
        'out': tf.Variable(tf.random_normal([output_dim])),
    }

    left = siamses_vgg_like_part(x1, weights, biases)
    right = siamses_vgg_like_part(x2, weights, biases)
    distance  = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(left, right),2),1,keep_dims=True))   
    loss = contrastive_loss(y, distance) 
    val_loss = contrastive_loss(y, distance) 
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    return x1, x2, y, left, right, distance, loss, val_loss, optimizer

def siamses_deep_share_part(x, weights, biases):
    dropout = 0.5
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv3 = maxpool2d(conv2, k=2)

    conv4 = conv2d(conv3, weights['wc3'], biases['bc3'])
    conv5 = conv2d(conv4, weights['wc4'], biases['bc4'])
    paddings = [[0, 0], [1, 0], [1, 0], [0, 0]]
    conv5 = tf.pad(conv5, paddings,mode='CONSTANT')
    conv6 = maxpool2d(conv5, k=2)

    conv7 = conv2d(conv6, weights['wc5'], biases['bc5'])
    paddings = [[0, 0], [1, 0], [0, 0], [0, 0]]
    conv7 = tf.pad(conv7, paddings,mode='CONSTANT')
    conv8 = maxpool2d(conv7, k=2)

    conv9 = conv2d(conv8, weights['wc6'], biases['bc6'])
    conv10 = maxpool2d(conv9, k=3)

    fc1 = tf.reshape(conv10, [-1, weights['out'].get_shape().as_list()[0]])
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
def siamses_deep(lr = 1e-3):
    """
    return a list of var
    """
    x1 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    x2 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    y = tf.placeholder(tf.float32, [None, 1])
    # Store layers weight & bias
    output_dim = config.CNN.output_dim
    weights = {
        # 3x3 conv, 1 input, 16 outputs
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 8])),

        'wc2': tf.Variable(tf.random_normal([3, 3, 8, 16])),

        'wc3': tf.Variable(tf.random_normal([3, 3, 16, 64])),

        'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64])),

        'wc5': tf.Variable(tf.random_normal([3, 3, 64, 64])),

        'wc6': tf.Variable(tf.random_normal([3, 3, 64, 64])),

        # fully connected, 7*7*64 inputs, 1024 outputs
        'out': tf.Variable(tf.random_normal([9 * 3 * 64, output_dim])),
        # 1024 inputs, 10 outputs (class prediction)
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([8])),
        'bc2': tf.Variable(tf.random_normal([16])),
        'bc3': tf.Variable(tf.random_normal([64])),
        'bc4': tf.Variable(tf.random_normal([64])),
        'bc5': tf.Variable(tf.random_normal([64])),
        'bc6': tf.Variable(tf.random_normal([64])),
        'out': tf.Variable(tf.random_normal([output_dim])),
    }

    left = siamses_deep_share_part(x1, weights, biases)
    right = siamses_deep_share_part(x2, weights, biases)
    distance  = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(left, right),2),1,keep_dims=True))   
    loss = contrastive_loss(y, distance) 
    val_loss = contrastive_loss(y, distance) 
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    return x1, x2, y, left, right, distance, loss, val_loss, optimizer

def siamses_test(lr=1e-3):
    """
    return a list of var
    """
    x1 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    x2 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    y = tf.placeholder(tf.float32, [None, 1])
    # Store layers weight & bias
    output_dim = config.CNN.output_dim
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'out': tf.Variable(tf.random_normal([21*7*64, output_dim])),
        # 1024 inputs, 10 outputs (class prediction)
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'out': tf.Variable(tf.random_normal([output_dim])),
    }
    left = siamses_test_share_part(x1, weights, biases)
    right = siamses_test_share_part(x2, weights, biases)
    distance  = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(left, right),2),1,keep_dims=True))   
    loss = contrastive_loss(y, distance) 
    val_loss = contrastive_loss(y, distance) 
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    return x1, x2, y, left, right, distance, loss, val_loss, optimizer

def get_accuracy(sess, dataset, x1, x2, left, right, distance, fake=False):
    # caculate reconition accuracy
    if fake is True:
        nm_view_2_accu = [{}, {}, {}]
        train_nm_accu = {"000":{"000":0.91, "180": 0}}
        test_nm_accu = {"018":{"054": 0.99, "180": 0.80}}
        nm_view_2_accu[0] = train_nm_accu
        nm_view_2_accu[1] = test_nm_accu
        return nm_view_2_accu

    conds = config.data.test_accu
    K = config.CNN.K
    correct_count = 0
    total_count = 0
    nm_view_2_accu = [{}, {}, {}]
    for cond_i, cond in enumerate(conds):
        for probe_view in ["%03d" % x for x in range(0, 181, 18)]:
            for gallery_view in ["%03d" % x for x in range(0, 181, 18)]:
                g_imgs, g_labels = dataset.get_gallerys(gallery_view)
                g_vectors = []
                g_vectors = sess.run(right, feed_dict={x2:g_imgs})
                for label in dataset.labels:
                    p_imgs = dataset.get_probes(label, probe_view, cond)
                    if len(p_imgs) == 0:
                        logging.warning("no probes of at label:%s, view:%s, cond:%s" % (label, probe_view, cond))
                        continue
                    p_vectors = sess.run(right, feed_dict={x2:p_imgs})
                    for p_v in p_vectors:
                        label_2_dists = {}
                        total_count += 1
                        for i in range(len(g_imgs)):
                            g_v = g_vectors[i]
                            g_l = g_labels[i]
                            left_val = p_v.reshape((1, p_v.shape[0]))
                            right_val = g_v.reshape((1, g_v.shape[0]))
                            d = sess.run(distance,\
                                feed_dict={\
                                    left:left_val,right:right_val})
                            d = d[0]
                            if g_l not in label_2_dists:
                                label_2_dists[g_l] = [d,]
                            else:
                                label_2_dists[g_l].append(d)
                        for l in label_2_dists.keys(): 
                            label_2_dists[l] = sorted(label_2_dists[l])[:K]
                        label_nearest_count = {}
                        for tmp in range(K):
                            min_dist = float("inf")
                            min_label = "no_label"
                            for l in label_2_dists.keys():
                                if len(label_2_dists[l]) > 0 and\
                                        label_2_dists[l][0] < min_dist:
                                    min_dist = label_2_dists[l][0]
                                    min_label = l
                            del label_2_dists[min_label][0]
                            if min_label not in label_nearest_count:
                                label_nearest_count[min_label] = 0
                            else:
                                label_nearest_count[min_label] += 1
                        max_count = 0
                        max_label = "no_label"
                        for l in label_nearest_count.keys():
                            if label_nearest_count[l] > max_count:
                                max_count = label_nearest_count[l]
                                max_label = l

                        if max_label == label:
                            correct_count += 1
                if total_count > 0:
                    accur = correct_count * 1.0 / total_count
                else:
                    accur = 0
                if probe_view not in nm_view_2_accu[cond_i]:
                    nm_view_2_accu[cond_i][probe_view] = {}
                nm_view_2_accu[cond_i][probe_view][gallery_view] = accur
    return nm_view_2_accu


if __name__ == '__main__':
    train_nm_accu = {"000":{"000":0.91, "180": 0}}
    test_nm_accu = {"018":{"054": 0.99, "180": 100}}
    d = collections.OrderedDict()
    d["tra nm"] = train_nm_accu
    d["tes nm"] =  test_nm_accu
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    output = output_res(d)
    logging.info(output)
