import tensorflow as tf
import logging
from tool.keras_tool import normalization_grey_image
import config
from tool.keras_tool import load_data
import os
import numpy as np
import config
import tf_models
from tf_models import get_accuracy

def main(data, val_data, test_data):
    model = getattr(tf_models, config.CNN.model_name)
    x1, x2, y, left, right, distance,loss,val_loss,optimizer=model(lr=config.CNN.lr)
    init = tf.initialize_all_variables()
    epoch = 2
    fragment_size = 512
    batch_count = 0
    val_every_batch = config.CNN.val_every

    logging.info("model name:%s,lr:%s, margin: %s, val_every:%s, K: %d" \
            % (config.CNN.model_name,config.CNN.lr\
                , config.CNN.margin, config.CNN.val_every\
                , config.CNN.K))

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

                    val_str = "nm  val\t"
                    #val_cl_str = "cl  val\t"
                    #val_bg_str = "bg  val\t"

                    nm_str = "nm test\t"
                    #cl_str = "cl test\t"
                    #bg_str = "bg test\t"

                    val_accu_sum = 0.0
                    #val_cl_accu_sum = 0.0
                    #val_bg_accu_sum = 0.0

                    nm_accu_sum = 0.0
                    #cl_accu_sum = 0.0
                    #bg_accu_sum = 0.0


                    for tmp in ["%03d" % x for x in range(0, 181, 18)]:
                        val_str += "%0.2f\t" % val_accu[tmp]
                        #val_cl_str += "%0.2f\t" % val_cl_accu[tmp]
                        #val_bg_str += "%0.2f\t" % val_bg_accu[tmp]

                        nm_str += "%0.2f\t" % nm_accu[tmp]
                        #cl_str += "%0.2f\t" % cl_accu[tmp]
                        #bg_str += "%0.2f\t" % bg_accu[tmp]

                        val_accu_sum += val_accu[tmp]
                        #val_cl_accu_sum += val_cl_accu[tmp]
                        #val_bg_accu_sum += val_bg_accu[tmp]

                        nm_accu_sum += nm_accu[tmp]
                        #cl_accu_sum += cl_accu[tmp]
                        #bg_accu_sum += bg_accu[tmp]

                    val_str += "%0.2f\t" % (val_accu_sum / 11.0)
                    #val_cl_str += "%0.2f\t" % (val_cl_accu_sum / 11.0)
                    #val_bg_str += "%0.2f\t" % (val_bg_accu_sum / 11.0)

                    nm_str += "%0.2f\t" % (nm_accu_sum / 11.0)
                    #cl_str += "%0.2f\t" % (cl_accu_sum / 11.0)
                    #bg_str += "%0.2f\t" % (bg_accu_sum / 11.0)

                    logging.info('\t'.join(["type :"] + ["%03d" % x for x in range(0, 181, 18)] + ['avg']))
                    logging.info(val_str)
                    logging.info(nm_str)
                    #logging.info(val_cl_str)
                    #logging.info(cl_str)
                    #logging.info(val_bg_str)
                    #logging.info(bg_str)

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
