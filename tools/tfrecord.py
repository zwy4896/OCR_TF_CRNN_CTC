'''
 * @Author: huan.wang 
 * @Date: 2019-04-04 17:45:27 
 * @Last Modified by:   huan.wang 
 * @Last Modified time: 2019-04-04 17:45:27 
''' 
import tensorflow as tf
import numpy as np
from PIL import Image
import os
# 真实样本 280宽度最长26个字符
# 真实样本 512宽度最长44个字符
PAD_TO = 512
TEXT_PAD_TO = 44
TEXT_PAD_VAL = 5989

class TFRecord_Reader(object):

    def parser(self, record):
        def dense_to_sparse(dense_tensor, sparse_val=0):
            with tf.name_scope("dense_to_sparse"):
                sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val),
                                       name="sparse_inds")
                sparse_vals = tf.gather_nd(dense_tensor, sparse_inds,
                                           name="sparse_vals")
                dense_shape = tf.shape(dense_tensor, name="dense_shape",
                                       out_type=tf.int64)
                return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)
        features = tf.parse_single_example(record,
                                       features={
                                           'images': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.VarLenFeature(tf.int64),
                                           'imagenames': tf.FixedLenFeature([], tf.string),
                                       })
        images = tf.image.decode_jpeg(features['images'])
        images.set_shape([32, None, 3])
        # 输入为灰度图像
        images = tf.image.rgb_to_grayscale(images)
        # pad to fixed number of bounding boxes
        pad_size = PAD_TO - tf.shape(images)[1]
        images = tf.pad(images, [[0, 0], [0, pad_size], [0, 0]], constant_values=255)
        images = tf.cast(images, tf.float32)
        images.set_shape([32, PAD_TO, 1])
        labels = tf.cast(features['labels'], tf.int32)
        labels_dense = labels.values
        pad_size = TEXT_PAD_TO - tf.shape(labels)[-1]
        labels_dense = tf.pad(labels_dense, [[0, pad_size]], constant_values=TEXT_PAD_VAL)
        labels = dense_to_sparse(labels_dense, sparse_val=-1)
        labels_length = tf.cast(tf.shape(labels)[-1], tf.int32)
        sequence_length = tf.cast(tf.shape(images)[-2] // 4, tf.int32)
        imagenames = features['imagenames']
        return images, labels, labels_dense, labels_length, sequence_length, imagenames

    def __init__(self, filenames, shuffle=True, batch_size=1):
        dataset = tf.data.TFRecordDataset(filenames)
        if shuffle:
            dataset = dataset.map(self.parser).repeat().batch(batch_size).shuffle(buffer_size=100)
        else:
            dataset = dataset.map(self.parser).repeat().batch(batch_size)
    
        self.iterator = dataset.make_one_shot_iterator()

    def read_and_decode(self):
        images, labels, labels_dense, labels_length, sequence_length, imagenames = self.iterator.get_next()
        return images, labels, labels_dense, labels_length, sequence_length, imagenames

if __name__ == '__main__':
    #train_f = '../densenet_ctc_synth300w_tfrecords/train.tfrecord'
    train_f = '/datacentre/huangkaijun/tfrecord_heng_test/train.tfrecord'
    tfrecord_reader = TFRecord_Reader([train_f], shuffle=True)
    vis = True
    init = tf.global_variables_initializer()
    images, _, labels_dense, _, _, _ = tfrecord_reader.read_and_decode()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        for i in range(1):
            image_, labels_dense_ = sess.run([images, labels_dense])
            print(labels_dense_)
            im = Image.fromarray(np.array(image_[0,:,:,0]), mode='L')
            im.save('tftest.jpg')
            if labels_dense_.shape[1]>20:
                print(labels_dense_.shape[1])
