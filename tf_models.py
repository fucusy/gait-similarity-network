import tensorflow as tf
import logging
from tool.keras_tool import normalization_grey_image
import config
from tool.keras_tool import load_data
import os
import numpy as np

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def contrastive_loss(y,d):
    margin = 200
    logging.info("set margin to %d" % margin)
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
def siamses_deep():
    """
    return a list of var
    """
    x1 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    x2 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    y = tf.placeholder(tf.float32, [None, 2])
    # Store layers weight & bias
    weights = {
        # 3x3 conv, 1 input, 16 outputs
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 8])),

        'wc2': tf.Variable(tf.random_normal([3, 3, 8, 16])),

        'wc3': tf.Variable(tf.random_normal([3, 3, 16, 64])),

        'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64])),

        'wc5': tf.Variable(tf.random_normal([3, 3, 64, 64])),

        'wc6': tf.Variable(tf.random_normal([3, 3, 64, 64])),

        # fully connected, 7*7*64 inputs, 1024 outputs
        'out': tf.Variable(tf.random_normal([9 * 3 * 64, 512])),
        # 1024 inputs, 10 outputs (class prediction)
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([8])),
        'bc2': tf.Variable(tf.random_normal([16])),
        'bc3': tf.Variable(tf.random_normal([64])),
        'bc4': tf.Variable(tf.random_normal([64])),
        'bc5': tf.Variable(tf.random_normal([64])),
        'bc6': tf.Variable(tf.random_normal([64])),
        'out': tf.Variable(tf.random_normal([512])),
    }

    left = siamses_deep_share_part(x1, weights, biases)
    right = siamses_deep_share_part(x2, weights, biases)
    distance  = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(left, right),2),1,keep_dims=True))   
    loss = contrastive_loss(y, distance) 
    val_loss = contrastive_loss(y, distance) 
    lr = 1e-3
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    return x1, x2, y, left, right, distance, loss, val_loss, optimizer
def siamses_test():
    """
    return a list of var
    """
    x1 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    x2 = tf.placeholder(tf.float32, [None, 210, 70, 1])
    y = tf.placeholder(tf.float32, [None, 2])
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'out': tf.Variable(tf.random_normal([21*7*64, 512])),
        # 1024 inputs, 10 outputs (class prediction)
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'out': tf.Variable(tf.random_normal([512])),
    }
    left = siamses_test_share_part(x1, weights, biases)
    right = siamses_test_share_part(x2, weights, biases)
    distance  = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(left, right),2),1,keep_dims=True))   
    loss = contrastive_loss(y, distance) 
    val_loss = contrastive_loss(y, distance) 
    lr = 1e-3
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    return x1, x2, y, left, right, distance, loss, val_loss, optimizer

def get_accuracy(sess, dataset, x1, x2, left, right, distance):
    # caculate reconition accuracy
    g_imgs, g_labels = dataset.get_gallerys()
    g_vectors = []
    g_vectors = sess.run(right, feed_dict={x2:g_imgs})
    nm_view_2_accu = [{}, {}, {}]
    conds = ['nm', 'cl', 'bg']
    for cond_i, cond in enumerate(conds):
        for probe_view in ["%03d" % x for x in range(0, 181, 18)]:
            correct_count = 0
            total_count = 0
            for label in dataset.labels:
                p_imgs = dataset.get_probes(label, probe_view, cond)
                p_vectors = sess.run(right, feed_dict={x2:p_imgs})
                for p_v in p_vectors:
                    min_dist = float("inf")
                    min_label = "no_label"
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
                        mean_dist = np.mean(np.array(label_2_dists[l]))
                        if mean_dist < min_dist:
                            min_dist = mean_dist
                            min_label = l
                    if min_label == label:
                        correct_count += 1
            if total_count > 0:
                accur = correct_count * 1.0 / total_count
            else:
                accur = 0
            nm_view_2_accu[cond_i][probe_view] = accur
    return nm_view_2_accu

def main(data, val_data, test_data):
    x1, x2, y, left, right, distance,loss,val_loss,optimizer=siamses_test()
    init = tf.initialize_all_variables()
    epoch = 2
    fragment_size = 512
    batch_count = 0
    val_every_batch = 100
    
    # visualization
    # op to write logs to path like 
    tmp = 0
    logs_path = './tensorflow_logs/%05d' % tmp
    while os.path.exists(logs_path):
        tmp += 1
        logs_path = './tensorflow_logs/%05d' % tmp
    os.makedirs(logs_path)
        

    summary_writer = tf.train.SummaryWriter(logs_path,graph=tf.get_default_graph())
    tf.scalar_summary("train loss", loss)
    tf.scalar_summary("validation loss", val_loss)
    merged_summary_op = tf.merge_all_summaries()
    print("Run the command line:\n" \
          "--> tensorboard --logdir=./tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
    with tf.Session() as sess:
        sess.run(init)
        data.reset_index()
        for i in range(epoch):
            while data.have_next():
                batch_count += 1
                batch_x, batch_y, _ = data.next_fragment(fragment_size, need_label=True)
                loss_val, summary = sess.run([loss, merged_summary_op], feed_dict={x1: batch_x[0], x2:batch_x[1], y: batch_y})
                logging.info("epoch %02d, batch count, %05d: Minibatch loss=%0.2f" % (i, batch_count, loss_val))
                summary_writer.add_summary(summary, batch_count)

                sess.run(optimizer, feed_dict={x1: batch_x[0], x2:batch_x[1], y: batch_y})
                if batch_count % val_every_batch == 0:
                    if not val_data.have_next():
                        val_data.reset_index()
                    batch_x, batch_y, _ = val_data.next_fragment(\
                                            fragment_size, need_label=True)
                    loss_val, summary = sess.run([val_loss, merged_summary_op], feed_dict={x1: batch_x[0], x2:batch_x[1], y: batch_y})
                    summary_writer.add_summary(summary, batch_count)
                    logging.info("epoch %02d, val loss=%0.2f" % (i, loss_val))
                    val_accu, val_cl_accu, val_bg_accu = get_accuracy(\
                            sess, val_data,x1, x2,left,right,distance)
                    nm_accu, cl_accu, bg_accu = get_accuracy(\
                            sess, test_data,x1, x2,left,right,distance)

                    val_str = "val\t"
                    test_str = "test\t"
                    val_accu_sum = 0.0
                    nm_accu_sum = 0.0
                    cl_accu_sum = 0.0
                    bg_accu_sum = 0.0


                    for tmp in ["%03d" % x for x in range(0, 181, 18)]:
                        val_str += "%0.2f\t" % val_accu[tmp]
                        nm_str += "%0.2f\t" % nm_accu[tmp]
                        cl_str += "%0.2f\t" % cl_accu[tmp]
                        bg_str += "%0.2f\t" % bg_accu[tmp]

                        val_accu_sum += val_accu[tmp]
                        nm_accu_sum += nm_accu[tmp]
                        cl_accu_sum += cl_accu[tmp]
                        bg_accu_sum += bg_accu[tmp]

                    val_str += "%0.2f\t" % (val_accu_sum / 11.0)
                    nm_str += "%0.2f\t" % (nm_accu_sum / 11.0)
                    cl_str += "%0.2f\t" % (cl_accu_sum / 11.0)
                    bg_str += "%0.2f\t" % (bg_accu_sum / 11.0)

                    logging.info('\t'.join(["type:"] + ["%03d" % x for x in range(0, 181, 18)] + ['avg']))
                    logging.info(val_str)
                    logging.info(nm_str)
                    logging.info(cl_str)
                    logging.info(bg_str)

if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    train_img_dirs = config.data.train_img_dirs 
    print train_img_dirs 
    train_data, validation_data, test_data = load_data(train_img_dirs)

    logging.info("train data image count %s" % train_data.count())
    logging.info("validation data image count %s" % validation_data.count())
    main(train_data, validation_data, test_data)
