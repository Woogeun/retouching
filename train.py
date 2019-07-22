from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import os
import random
import datetime
import cv2
from network import Network_
import glob

def mark_position_(elem, label, mode):
    level = len(elem)
    cur_level = 0
    pX = 0
    pY = 0
    for index in elem:
        if index == '2':
            pY += pow(2, 4 - cur_level)
        elif index == '3':
            pX += pow(2, 4 - cur_level)
        elif index == '4':
            pX += pow(2, 4 - cur_level)
            pY += pow(2, 4 - cur_level)


        cur_level += 1

    Xind = pX / pow(2, 5 - level)
    Yind = pY / pow(2, 5 - level)
    ind = Xind * pow(2, level - 1) + Yind
    base = 0
    for i in range(level-1):
        base += pow(4,i)
    label[base + int(ind)] = [1,0]

def parse_CTU_(ctu):
    elems = ctu.split()

    label = [[0,1] for i in range(85)]
    elems = [int(elem) for elem in elems]
    cur_level = 0

    level_index = [1 for i in range(5)]
    if len(elems)==1:
        return np.asarray(label)
    for elem in elems:
        if elem==99:
            cur_level = cur_level + 1
            prev_index = ''
            for i in range(cur_level):
                prev_index+=str(level_index[i])
            mark_position_(prev_index, label, cur_level)
            level_index[cur_level]=1
            continue
        else:
            if level_index[cur_level]==4:
                level_index[cur_level] = 1
                cur_level -= 1
                while(level_index[cur_level] ==4):
                    level_index[cur_level]=1
                    cur_level -= 1
                level_index[cur_level] += 1

            else:
                level_index[cur_level]+= 1
    return np.asarray(label)

def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    feature_description = {
        'double': tf.FixedLenFeature([], dtype=tf.string, default_value=""),
        'double_res': tf.FixedLenFeature([], dtype=tf.string, default_value=""),
        'single': tf.FixedLenFeature([], dtype=tf.string, default_value=""),
        'single_res': tf.FixedLenFeature([], dtype=tf.string, default_value=""),
        'dCTU': tf.FixedLenFeature([], dtype=tf.string, default_value=""),
        'sCTU': tf.FixedLenFeature([], dtype=tf.string, default_value=""),
        'label': tf.FixedLenFeature([], dtype=tf.string, default_value=""),
    }

    parsed_feature = tf.parse_single_example(example_proto, feature_description)
    single_img = tf.cast(tf.reshape(tf.decode_raw(parsed_feature['single'], tf.uint8), [64, 64, 3]), tf.float32)
    single_img_res = tf.cast(tf.reshape(tf.decode_raw(parsed_feature['single_res'], tf.uint8), [64, 64, 3]), tf.float32)
    double_img = tf.cast(tf.reshape(tf.decode_raw(parsed_feature['double'], tf.uint8), [64, 64, 3]), tf.float32)
    double_img_res = tf.cast(tf.reshape(tf.decode_raw(parsed_feature['double_res'], tf.uint8), [64, 64, 3]), tf.float32)

    sCTU = tf.cast(tf.reshape(tf.py_func(func=parse_CTU_, inp=[parsed_feature['sCTU']], Tout=tf.int32), [85,2]), tf.float32)
    sCTU_t = parsed_feature['sCTU']
    dCTU = tf.cast(tf.reshape(tf.py_func(func=parse_CTU_, inp=[parsed_feature['dCTU']], Tout=tf.int32), [85,2]), tf.float32)
    #dCTU = parsed_feature['dCTU']
    #s_label = tf.constant([1, 0], dtype= tf.uint8)
    s_label = tf.constant([1, 0, 0], dtype=tf.uint8)

    #d_label = tf.constant([0, 1], dtype=tf.uint8)
    d_label = tf.reshape(tf.io.decode_raw(parsed_feature['label'], tf.uint8), [3])
    return tf.concat([single_img, single_img_res], axis=-1), s_label, d_label, sCTU, dCTU, sCTU_t

batch_size = 100
files = tf.placeholder(dtype=tf.string)
train_files = glob.glob(r'H:\Videos\double\*\*\*\*\tfrecord\*.tfrecords')
random.shuffle(train_files)

test_files = glob.glob(r'H:\Videos\test\*\tfrecord\*.tfrecords')
dataset = tf.data.TFRecordDataset(files)
dataset = dataset.map(_parse_function, num_parallel_calls=8)
dataset = dataset.batch(batch_size)
dataset = dataset.shuffle(buffer_size=2000)
dataset = dataset.prefetch(buffer_size=2000)
dataset_iterator = dataset.make_initializable_iterator()
single_img, s_label, d_label, sCTU, dCTU, sCTU_t = dataset_iterator.get_next()





def operation_batch_seen(summary, curr_batch):
    global batch_seen
    global sum_interval_loss

    # print avg interval loss
    batch_seen += 1
    if batch_seen % loss_interval == 0 and batch_seen > 0:
        mean_loss = sum_interval_loss / loss_interval
        sum_interval_loss = 0
        log_string(LOG_FOUT_train, "batch seen: %d, curr batch : %d,  interval train loss: %s, interval: %d" % (batch_seen, curr_batch, mean_loss, loss_interval))


def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)



config = tf.ConfigProto(allow_soft_placement = True)
with tf.Session(config=config) as sess :

    loss_interval = 200
    save_interval = 100
    start_learning_rate = 1e-05
    lr_update_interval = 10000
    lr_update_rate = 0.95
    beta = 0.0001  # l2 normalizer parameter(preventing overfitting)
    Train = True
    log_path = './logs'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './logs/gradient_tape/' + current_time + '/train'
    test_log_dir = './logs/gradient_tape/' + current_time + '/test'
    weight_log_dir = './logs/gradient_tape/' + current_time + '/weight'
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(test_log_dir, exist_ok=True)
    os.makedirs(weight_log_dir, exist_ok=True)
    LOG_FOUT_train = open(train_log_dir + '/log_train_vgg.txt', "a")
    EPOCHS = 1000
    batch_seen = 0

    phase = tf.Variable(True, dtype=tf.bool)
    ce, acc = Network_(single_img, phase, sCTU)

    #ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=sCTU[:,4,:], logits=x_32)

    tf.summary.scalar('train/ce', tf.reduce_mean(ce) / 4)
    tf.summary.scalar('train/acc', tf.reduce_mean(acc)/ 4)
    merge = tf.summary.merge_all()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    vars = tf.global_variables()
    saver = tf.train.Saver(var_list=vars, write_version=tf.train.SaverDef.V2)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, lr_update_interval,
                                               lr_update_rate,
                                               staircase=True)
    with tf.control_dependencies(update_ops):
        loss = ce + sum(reg_losses)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    tf.global_variables_initializer().run()
    t_step = 0
    for epoch in range(EPOCHS):
        sess.run(dataset_iterator.initializer, feed_dict={files : train_files})
        log_string(LOG_FOUT_train, '**** Train EPOCH {} ****'.format(epoch))
        #a = sess.run(single_img)

        while True:
            try:

                _, summary = sess.run([train_op, merge], feed_dict={phase:True})

                t_step += 1
                if t_step % 10 == 0:
                    writer.add_summary(summary, t_step)
            except tf.errors.OutOfRangeError:
                break

        save_path = saver.save(sess, os.path.join(weight_log_dir, 'epoch' + str(epoch) + '_model.ckpt'))


        LOG_FOUT_test = open(test_log_dir + '/log_{}.txt'.format(epoch), "a")
        sess.run(dataset_iterator.initializer, feed_dict={files : test_files})
        log_string(LOG_FOUT_test, '**** Test EPOCH {} ****'.format(epoch))
        acc_ = 0
        step =0
        while True:
            try:
                accuracy = sess.run([acc], feed_dict={phase:False})
                acc_ += accuracy[0]
                step +=1

            except tf.errors.OutOfRangeError:
                break
        acc_ /= step
        log_string(LOG_FOUT_test, '{:.3f}'.format(acc_))
