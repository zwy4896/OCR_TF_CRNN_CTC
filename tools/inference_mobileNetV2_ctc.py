from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.getcwd()+'/')
# sys.path.append('~/wuyang.zhang/OCR_TF_CRNN_CTC')
import time
import json

import tensorflow as tf

import cv2
import numpy as np
import re
from PIL import Image

slim = tf.contrib.slim

from nets import densenet
from nets.cnn.dense_net import DenseNet
from nets.cnn.mobile_net_v2 import MobileNetV2

# keys有变化
import keys_old as keys
# import keys

os.environ["CUDA_VISIBLE_DEVICES"]=""

_IMAGE_HEIGHT = 32

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_string(
    'image_dir', '/tmp/chn/', 'Path to the directory containing images.')
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/multigpu_model/', 'Base directory for the model.')

FLAGS = tf.app.flags.FLAGS

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)
char_map_dict = {}
for i, val in enumerate(characters):
    char_map_dict[val] = i

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.image_dir):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def get_labels():
    labels = {}
    # hehe测试集ground truth，result_hehe.txt
    # with open(os.path.join(FLAGS.image_dir, 'result_hehe.txt'), encoding='utf8') as f:
    # 普通测试集，labels.txt
    with open(os.path.join(FLAGS.image_dir, 'labels.txt'), encoding='utf8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # hehe测试集，以'卍'符号分割
            labels[os.path.join(FLAGS.image_dir, line.split('卍')[0])]= line.split('卍')[1]
            # 普通测试集，以空格分隔
            # labels[os.path.join(FLAGS.image_dir, line.split()[0])]= line.split()[1]

    return labels

def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims
  
def _sparse_matrix_to_list(sparse_matrix, char_map_dict=None):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    # the last index in sparse_matrix is ctc blanck note
    if char_map_dict is None:
        char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')    

    dense_matrix =  len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(_int_to_string(val, char_map_dict))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list

def _int_to_string(value, char_map_dict=None):
    if char_map_dict is None:
        char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')
    
    return(characters[value])
    '''
    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return "" 
    raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))
    '''

def _LSTM_cell(num_proj=None):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=256, num_proj=num_proj)
    return cell

def _bidirectional_LSTM(inputs, num_out, seq_len):
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(_LSTM_cell(),
                                                _LSTM_cell(),
                                                inputs,
                                                sequence_length=seq_len,
                                                dtype=tf.float32)

    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [-1, 256 * 2])

    outputs = slim.fully_connected(outputs, num_out, activation_fn=None)

    shape = tf.shape(inputs)
    outputs = tf.reshape(outputs, [shape[0], -1, num_out])

    return outputs
def _inference_densenet_ctc():
    '''
    temp = "想做/ 兼_职/学生_/ 的 、加,我Q：  1 5.  8 0. ！！？？  8 6 。0.  2。 3     有,惊,喜,哦"
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#：￥%……&*（）]+", "",temp)
    print(string)
    '''

    input_image = tf.placeholder(dtype=tf.float32, shape=[1, _IMAGE_HEIGHT, None, 1])

    '''
    # initialise the net model
    with slim.arg_scope(densenet.densenet_arg_scope(weight_decay=0.0004)):
            with tf.variable_scope('DENSENET_CTC', reuse=False):
                first_output_features = 64
                layers_per_block = 8
                growth_rate = 8
                net, _ = densenet.densenet_40(input_image, 5990, first_output_features, layers_per_block, growth_rate, is_training = False)
    '''

    with tf.variable_scope('MobilenetV2', reuse=False):
        net = MobileNetV2(input_image, is_training=False)
        cnn_out = net.net

        cnn_output_shape = tf.shape(cnn_out)
        batch_size = cnn_output_shape[0]
        cnn_output_h = cnn_output_shape[1]
        cnn_output_w = cnn_output_shape[2]
        cnn_output_channel = cnn_output_shape[3]

        # Get seq_len according to cnn output, so we don't need to input this as a placeholder
        seq_len = tf.ones([batch_size], tf.int32) * cnn_output_w

        # Reshape to the shape lstm needed. [batch_size, max_time, ..]
        cnn_out_transposed = tf.transpose(cnn_out, [0, 2, 1, 3])
        cnn_out_reshaped = tf.reshape(cnn_out_transposed, [batch_size, cnn_output_w, cnn_output_h * cnn_output_channel])

        cnn_shape = cnn_out.get_shape().as_list()
        cnn_out_reshaped.set_shape([None, cnn_shape[2], cnn_shape[1] * cnn_shape[3]])
        
        '''
        bilstm = cnn_out_reshaped
        for i in range(2):
            with tf.variable_scope('bilstm_%d' % (i + 1)):
                if i == 1:
                    bilstm = _bidirectional_LSTM(bilstm, 5990, seq_len)
                else:
                    bilstm = _bidirectional_LSTM(bilstm, 256, seq_len)

        logits = bilstm
        '''

        logits = slim.fully_connected(cnn_out_reshaped, 5990, activation_fn=None)
        # logits = slim.fully_connected(cnn_out_reshaped, 6049, activation_fn=None)
        # logits = tf.layers.dense(cnn_out_reshaped, 5990)



        # ctc require time major
        logits = tf.transpose(logits, (1, 0, 2))

    #input_sequence_length = tf.placeholder(tf.int32, shape=[1], name='input_sequence_length')

    ctc_decoded, ctc_log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=True)

    image_names = get_images()
    gt_labels = get_labels()

    # set checkpoint saver
    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(FLAGS.model_dir)

    with tf.Session() as sess:
        # restore all variables
        saver.restore(sess=sess, save_path=save_path)

        accuracy = []

        for image_name in image_names:
            # print(image_name)
            image_path = image_name
            image = cv2.imread(image_path)
            # image = np.rot90(image)
            # print(image.shape)
            h, w, c = image.shape
            height = _IMAGE_HEIGHT
            width = int(w * height / h)
            # print(height, width)
            if width > height:
                print('ok')
                image = cv2.resize(image, (width, height))[:,:,::-1]
                image = Image.fromarray(image).convert('L')
                new_image = np.zeros((32, width))
                new_image[:,0:width] = image
                image = np.expand_dims(new_image, axis=0)
                image = np.expand_dims(image, axis=-1)
                image = np.array(image, dtype=np.float32)

                #seq_len = np.array([width / 8], dtype=np.int32)

                preds = sess.run(ctc_decoded, feed_dict={input_image:image})
    
                preds = _sparse_matrix_to_list(preds[0], char_map_dict)


                print('Predict {:s} image as: {:s}'.format(image_name, preds[0]))

                pred = preds[0]
                gt_label = gt_labels[image_name]
                pred = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！〔!，。《》；;'‘’<>【】/?？、·°""“”~@#：④|:￥%……&*()（）]+", "",pred)
                gt_label = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！〔!，。《》；;'‘’<>【】/?？、·°""“”~@#：④|:￥%……&*()（）]+", "",gt_label)
                total_count = len(gt_label)
                correct_count = 0
                # with open('result_heng_test.txt', 'a+', encoding='utf-8') as f:
                #     f.write(image_name.split('/')[-1] + '卍' + pred + '\n')
                try:
                    for i, tmp in enumerate(gt_label):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / total_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)
        accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
        print('accuracy={:9f}'.format(accuracy))

        
def main(unused_argv):
    _inference_densenet_ctc()

if __name__ == '__main__':
    tf.app.run() 
