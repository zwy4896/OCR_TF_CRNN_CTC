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
PAD_TO = 768
TEXT_PAD_TO = 80
TEXT_PAD_VAL = 4304
# PAD_TO = 280
# TEXT_PAD_TO = 60
# TEXT_PAD_VAL = 93

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
        # features = tf.parse_single_example(record,
        #                        features={
        #                            'image/encoded': tf.FixedLenFeature([], tf.string),
        #                            'image/labels': tf.VarLenFeature(tf.int64),
        #                            'image/filename': tf.FixedLenFeature([], tf.string),
        #                        })
        images = tf.image.decode_jpeg(features['images'])
        # images = tf.image.decode_jpeg(features['image/encoded'])

        # print(images.shape)
        images.set_shape([32, None, 3])
        # 输入为灰度图像
        images = tf.image.rgb_to_grayscale(images)
        # print(images.shape)
        # pad to fixed number of bounding boxes
        pad_size = PAD_TO - tf.shape(images)[1]
        images = tf.pad(images, [[0, 0], [0, pad_size], [0, 0]], constant_values=255)
        # images = tf.image.resize_images(images, (32, PAD_TO))
        images = tf.cast(images, tf.float32)
        images.set_shape([32, PAD_TO, 1])
        labels = tf.cast(features['labels'], tf.int32)
        # labels = tf.cast(features['image/labels'], tf.int32)
        labels_dense = labels.values
        pad_size = TEXT_PAD_TO - tf.shape(labels)[-1]
        # print(labels_dense.shape)
        labels_dense = tf.pad(labels_dense, [[0, pad_size]], constant_values=TEXT_PAD_VAL)
        labels = dense_to_sparse(labels_dense, sparse_val=-1)
        labels_length = tf.cast(tf.shape(labels)[-1], tf.int32)
        # print(labels_length.shape)
        sequence_length = tf.cast(tf.shape(images)[-2] // 4, tf.int32)
        imagenames = features['imagenames']
        # imagenames = features['image/filename']

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
    # train_f = '/data01/wuyang.zhang/github/cnn_lstm_ctc_ocr/data/train/words-000.tfrecord'
    train_f = '/algdata01/huangkaijun/data_all/tfrecord_768_fanti/tfrecord/train.tfrecord'
    tfrecord_reader = TFRecord_Reader([train_f], shuffle = True)
    init = tf.global_variables_initializer()
    images, _, labels_dense, _, _, imagenames = tfrecord_reader.read_and_decode()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        c = 0
        width = []
        for i in range(1):
            image_, labels_dense_, names = sess.run([images, labels_dense, imagenames])
            print(labels_dense_)
            # print(image_.shape)

            im = Image.fromarray(np.array(image_[0,:,:,0]), mode = 'L')
            # width.append(im.size[0])

            print(im.size)
            # im = Image.fromarray(np.array(image_), mode='L')
            im.save('tftest1.jpg')
            # if 270 < im.size[0] < 300:
                # im.save('tftest' + str(i) + '.jpg')
            # print(labels_dense_)
            # print(names)
        #     c+= 1
        # print(c)
        # print(max(width))

