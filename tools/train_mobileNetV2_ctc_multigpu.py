from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.getcwd()+'/')
sys.path.append('/tmp/workspace/OCR_TF_CRNN_CTC')
import time
import json

import tensorflow as tf
slim = tf.contrib.slim

import numpy as np

from nets import densenet
from nets.cnn.dense_net import DenseNet
from nets.cnn.paper_cnn import PaperCNN
from nets.cnn.mobile_net_v2 import MobileNetV2
import keys_old as keys    # keys.py为6049字符，keys_old中为5990个字符
#import warpctc_tensorflow
from tfrecord import TFRecord_Reader

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ------------------------------------Basic prameters------------------------------------
#tf.app.flags.DEFINE_string(
#    'data_dir', '../densenet_ctc_synth300w_tfrecords/', 'Path to the directory containing data tf record.')
tf.app.flags.DEFINE_string(
    'data_dir', '/datacentre/huan.wang/densenet_ctc_synth300w_tfrecords', 'Path to the directory containing data tf record.')

tf.app.flags.DEFINE_boolean('restore', True, 'whether to resotre from checkpoint')   

tf.app.flags.DEFINE_string('gpu_list', '2,3', '')

tf.app.flags.DEFINE_string(
    'model_dir', 'MobileNetV2_ckpt_20190719/', 'Base directory for the model.')

tf.app.flags.DEFINE_integer(
    'num_threads', 1, 'The number of threads to use in batch shuffling') 

tf.app.flags.DEFINE_integer(
    'step_per_save', 1000, 'The number of training steps to run between save checkpoints.')

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_integer(
    'batch_size', 64, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_train_steps', 200000, 'The number of maximum iteration steps for training')

tf.app.flags.DEFINE_float(
    'learning_rate', 0.0005, 'The initial learning rate for training.') #0.0005

tf.app.flags.DEFINE_integer(
    'decay_steps', 10000, 'The learning rate decay steps for training.')

tf.app.flags.DEFINE_float(
    'decay_rate', 0.632, 'The learning rate decay rate for training.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)
char_map_dict = {}
for i, val in enumerate(characters):
    char_map_dict[val] = i

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
    
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

def _LSTM_cell(num_proj=None):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=256, num_proj=num_proj)
    #cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob= 1.0)
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

def tower_loss(input_images, input_labels, input_labels_dense, input_labels_lengths, input_sequence_lengths, reuse_variables=None):
    # Build inference graph
    # initialise the net model
    '''
    with slim.arg_scope(densenet.densenet_arg_scope(weight_decay=0.0004)):
        with tf.variable_scope('DENSENET_CTC', reuse=reuse_variables):
            first_output_features = 64
            layers_per_block = 8
            growth_rate = 8
            net, _ = densenet.densenet_40(input_images, 5990, first_output_features, layers_per_block, growth_rate, is_training = True)
            
        ctc_loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=input_labels, inputs=net, sequence_length=input_sequence_lengths,
                ignore_longer_outputs_than_inputs=True))
        #ctc_loss = tf.reduce_mean(warpctc_tensorflow.ctc(net, tf.reshape(input_labels_dense, [-1]), \
        #    tf.reshape(input_labels_lengths, [-1]), tf.reshape(input_sequence_lengths, [-1])))
        total_loss = tf.add_n([ctc_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    '''
    with tf.variable_scope('MobilenetV2', reuse=reuse_variables):
        net = MobileNetV2(input_images, is_training=True)

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
        logits = slim.fully_connected(cnn_out_reshaped, 5990, activation_fn = None)

        # ctc require time major
        logits = tf.transpose(logits, (1, 0, 2))

        ctc_loss = tf.reduce_mean(
                tf.nn.ctc_loss(labels=input_labels, inputs=logits, sequence_length=seq_len,
                    ignore_longer_outputs_than_inputs=True))
        #ctc_loss = tf.reduce_mean(warpctc_tensorflow.ctc(net, tf.reshape(input_labels_dense, [-1]), \
        #    tf.reshape(input_labels_lengths, [-1]), tf.reshape(input_sequence_lengths, [-1])))
        total_loss = tf.add_n([ctc_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # add summary
        if reuse_variables is None:
            tf.summary.scalar(name='CTC_Loss', tensor=ctc_loss)
            tf.summary.scalar(name='TOTAL_Loss', tensor=total_loss)
  
    return total_loss, ctc_loss

def _train_densenetocr_ctc():
    # tfrecord_path = [os.path.join(FLAGS.data_dir, 'train20190707.tfrecord'), os.path.join(FLAGS.data_dir, 'train32_768_20190707.tfrecord'), os.path.join(FLAGS.data_dir, 'train.tfrecord'), os.path.join(FLAGS.data_dir, 'train_02.tfrecord'), os.path.join(FLAGS.data_dir, 'validation.tfrecord')]
    tfrecord_path = ['/datacentre/wuyang.zhang/300w_tfrecords_32_512_sin_blur_20190703/train.tfrecord']
    tfrecord_path.append(os.path.join(FLAGS.data_dir, 'train.tfrecord'))
    tfrecord_path.append('/datacentre/wuyang.zhang/real_tfrecords_20190712/train_32_280_20190712.tfrecord')
    tfrecord_path.append('/datacentre/wuyang.zhang/real_imgs_tfrecord_32_280_512_20190704/train_real_32_280_512.tfrecord')

    tfrecord_reader = TFRecord_Reader(tfrecord_path, batch_size=FLAGS.batch_size)
    batch_images, batch_labels, batch_labels_dense, batch_input_labels_lengths, batch_sequence_lengths, _ = tfrecord_reader.read_and_decode()

    global_step = tf.train.create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)

    # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # split
    input_images_split = tf.split(batch_images, len(gpus))
    input_labels_split = tf.sparse.split(sp_input=batch_labels, num_split=len(gpus), axis=0)
    input_labels_dense_split = tf.split(batch_labels_dense, len(gpus))
    input_labels_lengths_split = tf.split(batch_input_labels_lengths, len(gpus))
    input_sequence_lengths_split = tf.split(batch_sequence_lengths, len(gpus))

    tower_grads = []
    reuse_variables = None

    tf.ConfigProto(allow_soft_placement=True)
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                ilms = input_labels_split[i]
                ildms = input_labels_dense_split[i]
                ilsms = input_labels_lengths_split[i]
                iisms = input_sequence_lengths_split[i]
                total_loss, ctc_loss = \
                    tower_loss(iis, ilms, ildms, ilsms, iisms, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = optimizer.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, tf.get_default_graph())

    init_op = tf.global_variables_initializer()

    # set checkpoint saver
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'crnn_ctc_ocr_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(FLAGS.model_dir, model_name)  

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
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

        for step in range(FLAGS.max_train_steps):
            
            lr, cl, tl, _ = sess.run([learning_rate, ctc_loss, total_loss, train_op])
            
            print('step:{:d} learning_rate={:9f} ctc_loss={:9f} total_loss={:9f}'.format(
                    step + 1, lr, cl, tl))

            if (step + 1) % FLAGS.step_per_save == 0: 
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary=summary_str, global_step=step)
                saver.save(sess=sess, save_path=model_save_path, global_step=step)
            
        # close tensorboard writer
        summary_writer.close()

def main(unused_argv):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    _train_densenetocr_ctc()

if __name__ == '__main__':
    tf.app.run() 
