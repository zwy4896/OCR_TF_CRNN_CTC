from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.getcwd()+'/')
import time
import json

import tensorflow as tf
slim = tf.contrib.slim

import numpy as np

from nets import densenet
from nets.cnn.dense_net import DenseNet

import keys
import warpctc_tensorflow
from tfrecord import TFRecord_Reader

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_string(
    'data_dir', '/algdata01/wuyang.zhang/300w_tfrecords_32_512_sin_blur_20190703/', 'Path to the directory containing data tf record.')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')   

tf.app.flags.DEFINE_string(
    'model_dir', './model/', 'Base directory for the model.')

tf.app.flags.DEFINE_integer(
    'num_threads', 1, 'The number of threads to use in batch shuffling') 

tf.app.flags.DEFINE_integer(
    'step_per_eval', 1, 'The number of training steps to run between evaluations.')

tf.app.flags.DEFINE_integer(
    'step_per_save', 1000, 'The number of training steps to run between save checkpoints.')

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_train_steps', 20000, 'The number of maximum iteration steps for training')

tf.app.flags.DEFINE_float(
    'learning_rate', 0.0005, 'The initial learning rate for training.')

tf.app.flags.DEFINE_integer(
    'decay_steps', 10000, 'The learning rate decay steps for training.')

tf.app.flags.DEFINE_float(
    'decay_rate', 0.652, 'The learning rate decay rate for training.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')

FLAGS = tf.app.flags.FLAGS

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)
char_map_dict = {}
for i, val in enumerate(characters):
    char_map_dict[val] = i
  
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

def _train_densenetocr_ctc():
    tfrecord_path = os.path.join(FLAGS.data_dir, 'train_32_512_20190723.tfrecord')

    tfrecord_reader = TFRecord_Reader([tfrecord_path], batch_size=FLAGS.batch_size)
    batch_images, batch_labels, batch_labels_dense, batch_input_labels_lengths, batch_sequence_lengths, _ = tfrecord_reader.read_and_decode()

    input_images = tf.placeholder(tf.float32, shape=[128, 32, 512, 1], name='input_images')
    input_labels = tf.sparse_placeholder(tf.int32, name='input_labels')
    input_labels_dense = tf.placeholder(tf.int32, shape=[None], name='input_labels_dense')
    input_labels_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='input_labels_lengths')
    input_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='input_sequence_lengths')

    #char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    # initialise the net model
    # with slim.arg_scope(densenet.densenet_arg_scope(weight_decay=0.0004)):
    #     with tf.variable_scope('DENSENET_CTC', reuse=False):
    #         first_output_features = 64
    #         layers_per_block = 8
    #         growth_rate = 8
    #         net, _ = densenet.densenet_40(input_images, 5990, first_output_features, layers_per_block, growth_rate, is_training = True)
    with tf.variable_scope('DENSENET_CTC', reuse=None):
            net = DenseNet(input_images, is_training=True)


    # ctc_loss = tf.reduce_mean(
    #    tf.nn.ctc_loss(labels=input_labels, inputs=net, sequence_length=input_sequence_lengths,
    #        ignore_longer_outputs_than_inputs=True))
    ctc_loss = tf.reduce_mean(warpctc_tensorflow.ctc(net, tf.reshape(input_labels_dense, [-1]), \
        tf.reshape(input_labels_lengths, [-1]), tf.reshape(input_sequence_lengths, [-1])))

    #ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net, input_sequence_lengths, merge_repeated=False)
    ctc_decoded, ct_log_prob = tf.nn.ctc_greedy_decoder(net, input_sequence_lengths, merge_repeated=False)

    #sequence_distance = tf.reduce_mean(tf.edit_distance(tf.cast(ctc_decoded[0], tf.int32), input_labels))

    global_step = tf.train.create_global_step()

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)

    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=ctc_loss, global_step=global_step)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=ctc_loss, global_step=global_step)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss=ctc_loss, global_step=global_step)
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, optimizer, update_ops]):
        train_op = tf.no_op(name='train_op')
        
    init_op = tf.global_variables_initializer()

    # set tf summary
    tf.summary.scalar(name='CTC_Loss', tensor=ctc_loss)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    #tf.summary.scalar(name='Seqence_Distance', tensor=sequence_distance)
    merge_summary_op = tf.summary.merge_all()

    # set checkpoint saver
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'crnn_ctc_ocr_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(FLAGS.model_dir, model_name)  

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.model_dir)
        summary_writer.add_graph(sess.graph)

        # init all variables
        sess.run(init_op)

        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
            variable_restore_op = slim.assign_from_checkpoint_fn(ckpt, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
            #saver.restore(sess, ckpt)
            variable_restore_op(sess)

        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        concat_imgs = []
        concat_lbls = []
        concat_lbls_dense = []
        concat_lbls_lens = []
        concat_seq_lens = []
     
        for step in range(FLAGS.max_train_steps):
            imgs, lbls, lbls_dense, lbls_lens, seq_lens = sess.run([batch_images, batch_labels, batch_labels_dense, batch_input_labels_lengths, batch_sequence_lengths])
            #print(type(lbls_dense),lbls_dense.shape)
            concat_imgs.append(imgs[0])
            concat_lbls_dense.append(lbls_dense[0])
            concat_lbls_lens.append(lbls_lens[0])
            concat_seq_lens.append(seq_lens[0])
            if (step+1) % 128 == 0:
                imgs = np.array(concat_imgs)
                lbls_dense = np.hstack(concat_lbls_dense)
                lbls_lens = np.hstack(concat_lbls_lens)
                seq_lens = np.hstack(concat_seq_lens)
                print(imgs.shape, lbls_dense.shape, lbls_lens.shape, seq_lens.shape)
                
                _, cl, lr, summary = sess.run(
                    [train_op, ctc_loss, learning_rate, merge_summary_op],
                    feed_dict = {input_images:imgs, input_labels_dense:np.reshape(lbls_dense,[-1]), input_labels_lengths:np.reshape(lbls_lens,[-1]), input_sequence_lengths:np.reshape(seq_lens,[-1])})
            #sess.run(
            #    [net_out],
            #    feed_dict = {input_images:imgs, input_labels:lbls, input_sequence_lengths:seq_lens})
            
                print('step:{:d} learning_rate={:9f} ctc_loss={:9f} '.format(
                        step + 1, lr, cl))

                concat_imgs = []
                concat_lbls = []
                concat_lbls_dense = []
                concat_lbls_lens = []
                concat_seq_lens = []

            if (step + 1) % FLAGS.step_per_save == 0: 
                summary_writer.add_summary(summary=summary, global_step=step)
                saver.save(sess=sess, save_path=model_save_path, global_step=step)
            '''
            if (step + 1) % FLAGS.step_per_eval == 0:
                _, cl, lr, sd, preds, summary = sess.run(
                [train_op, ctc_loss, learning_rate, sequence_distance, ctc_decoded, merge_summary_op],
                feed_dict = {input_images:imgs, input_labels:lbls, input_labels_dense:np.reshape(lbls_dense,[-1]), input_labels_lengths:np.reshape(lbls_lens,[-1]), input_sequence_lengths:np.reshape(seq_lens,[-1])})

                # calculate the precision
                preds = _sparse_matrix_to_list(preds[0], char_map_dict)
                gt_labels = _sparse_matrix_to_list(lbls, char_map_dict)

                accuracy = []

                for index, gt_label in enumerate(gt_labels):
                    pred = preds[index]
                    total_count = len(gt_label)
                    correct_count = 0
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

                print('step:{:d} learning_rate={:9f} ctc_loss={:9f} sequence_distance={:9f} train_accuracy={:9f}'.format(
                    step + 1, lr, cl, sd, accuracy))
            '''
        # close tensorboard writer
        summary_writer.close()

        # stop file queue
        #coord.request_stop()
        #coord.join(threads=threads)

def main(unused_argv):
    _train_densenetocr_ctc()

if __name__ == '__main__':
    tf.app.run() 
