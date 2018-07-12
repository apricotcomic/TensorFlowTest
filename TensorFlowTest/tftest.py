import random
import numpy as np
import cv2
import tensorflow as tf


CHANNELS = 3
NUM_CLASSES = 3
IMAGE_SIZE = 28
IMAGE_MATRIX_SIZE = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
PATH_LABEL_FILE = "path_and_label.txt"


PATH_AND_LABEL = []

with open(PATH_LABEL_FILE, mode='r') as file :

    for line in file :
        # 改行を除く
        line = line.rstrip()
        # スペースで区切られたlineを、配列にする
        line_list = line.split()
        PATH_AND_LABEL.append(line_list)
        random.shuffle(PATH_AND_LABEL)


DATA_SET = []

for path_label in PATH_AND_LABEL :

    tmp_list = []

    # 画像を読み込み、サイズを変更する
    img = cv2.imread(path_label[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    # 行列を一次元に、dtypeをfloat32に、０～１の値に正規化する
    img = img.flatten().astype(np.float32)/255.0

    tmp_list.append(img)

    # 分類するクラス数の長さを持つ仮のリストを作成する
    classes_array = np.zeros(NUM_CLASSES, dtype = 'float64')
    # ラベルの数字によって、リストを更新する
    classes_array[int(path_label[1])] = 1

    tmp_list.append(classes_array)

    DATA_SET.append(tmp_list)


TRAIN_DATA_SIZE = int(len(DATA_SET) * 0.8)
TRAIN_DATA_SET = DATA_SET[:TRAIN_DATA_SIZE]
TEST_DATA_SET = DATA_SET[TRAIN_DATA_SIZE:]


def batch_data(data_set, batch_size) :

    data_set = random.sample(data_set, batch_size)

    return data_set


def devide_data_set(data_set) :
    data_set = np.array(data_set)
    image_data_set = data_set[:int(len(data_set)), :1].flatten()
    label_data_set = data_set[:int(len(data_set)), 1:].flatten()

    image_ndarray = np.empty((0, IMAGE_MATRIX_SIZE))
    label_ndarray = np.empty((0, NUM_CLASSES))

    for (img, label) in zip(image_data_set, label_data_set) :
        image_ndarray = np.append(image_ndarray, np.reshape(img, (1, IMAGE_MATRIX_SIZE)), axis=0)
        label_ndarray = np.append(label_ndarray, np.reshape(label, (1, NUM_CLASSES)), axis=0)

    return image_ndarray, label_ndarray


def conv2d(x, W) :
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x) :
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape) :
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape) :
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def deepnn(x) :

    with tf.name_scope('reshape') :
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

    with tf.name_scope('conv1') :
        W_conv1 = weight_variable([5, 5, CHANNELS, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') :
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') :
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') :
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') :
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout') :
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') :
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def main(_):

    x = tf.placeholder(tf.float32, [None, IMAGE_MATRIX_SIZE])

    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        for epoch_step in range(MAX_EPOCH) :
            train_data_set = batch_data(TRAIN_DATA_SET, BATCH_SIZE)
            train_image, train_label = devide_data_set(train_data_set)

            if epoch_step % BATCH_SIZE == 0 :
                train_accuracy = accuracy.eval(feed_dict={x: train_image, y_: train_label, keep_prob: 1.0})
                print('epoch_step %d, training accuracy %g' % (epoch_step, train_accuracy))

            train_step.run(feed_dict={x: train_image, y_: train_label, keep_prob: 0.5})

        test_image, test_label = devide_data_set(TEST_DATA_SET)
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_image, y_: test_label, keep_prob: 1.0}))


MAX_EPOCH = 1000
BATCH_SIZE = 50


if __name__ == '__main__' :
    main(_)

epoch_step ~, training accuracy ~
test accuracy ~

