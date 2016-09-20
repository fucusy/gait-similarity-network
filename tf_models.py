import tensorflow as tf
import logging
from tool.keras_tool import normalization_grey_image
import config
from tool.keras_tool import load_data

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def contrastive_loss(y,d):
    part1 = y * tf.square(d)
    part2 = (1-y) * tf.square(tf.maximum((1 - d),0))
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

def siamses_test(data, val_data):
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
    lr = 1e-3
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.initialize_all_variables()
    epoch = 2
    fragment_size = 512
    batch_count = 0
    val_every_batch = 50
    with tf.Session() as sess:
        sess.run(init)
        data.reset_index()
        for i in range(epoch):
            while data.have_next():
                batch_count += 1
                batch_x, batch_y, _ = data.next_fragment(fragment_size, need_label=True, preprocess_fuc=normalization_grey_image)
                loss_val = sess.run(loss, feed_dict={x1: batch_x[0], x2:batch_x[1], y: batch_y})
                print("epoch %02d, batch count, %05d: Minibatch loss=%0.2f" % (epoch, batch_count, loss_val))

                sess.run(optimizer, feed_dict={x1: batch_x[0], x2:batch_x[1], y: batch_y})
                if batch_count % val_every_batch == 0:
                    if not val_data.have_next():
                        val_data.reset_index()
                    batch_x, batch_y, _ = val_data.next_fragment(fragment_size, need_label=True, preprocess_fuc=normalization_grey_image)
                    loss_val = sess.run(loss, feed_dict={x1: batch_x[0], x2:batch_x[1], y: batch_y})
                    print("epoch %02d, val loss=%0.2f" % (i, loss_val))
if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    train_img_dirs = config.data.train_img_dirs 
    print train_img_dirs 
    train_data, validation_data = load_data(train_img_dirs)

    logging.info("train data image count %s" % train_data.count())
    logging.info("validation data image count %s" % validation_data.count())
    siamses_test(train_data, validation_data)
