import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
from collections import namedtuple
# from keras.layers import DepthwiseConv2D

DensenetParams = namedtuple('DensenetParameters', ['first_output_features',
                                                   'layers_per_block',
                                                   'growth_rate',
                                                   'bc_mode',
                                                   'dropout_keep_prob'
                                                   ])

weight_decay = 5e-7
def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def global_avg(x,s=1):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], s)
        return net

def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)

def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]

        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        # conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        conv = slim.separable_conv2d(input, 
            None,
            [k_h, k_w],
            depth_multiplier=1,
            stride=1,
            rate=1,
            normalizer_fn=None,
            padding=padding,)
        # print(conv.shape)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv

def squeeze_excitation_layer(input, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :

        squeeze = global_avg(input)
        squeeze = flatten(squeeze)
        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_excitation1')
        excitation = tf.nn.relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_excitation2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input * excitation

        return scale

class DenseNet(object):
    default_params = DensenetParams(
        first_output_features=16,
        layers_per_block=8,
        growth_rate=8,
        bc_mode=True,
        dropout_keep_prob=0.8,
    )

    def __init__(self, inputs, params=None, is_training=True):
        if isinstance(params, DensenetParams):
            self.params = params
        else:
            self.params = DenseNet.default_params

        self._scope = 'densenet'

        self.is_training = is_training
        with slim.arg_scope(self.arg_scope(is_training)):
            self.build_net(inputs)
        
    def build_net(self, inputs):
        num_channels = self.params.first_output_features

        with tf.variable_scope(self._scope, self._scope, [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'

            with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.fully_connected],
                                outputs_collections=end_points_collection):
                with tf.variable_scope("conv_1"):
                    print('inputs: ', inputs.shape)
                    # print(num_channels)
                    net = slim.conv2d(inputs, self.params.first_output_features, [3, 3])
                    # net = dwise_conv(inputs, channel_multiplier=num_channels)
                    print('conv1: ', net.shape)

                with tf.variable_scope('dense_block_1'):
                    # add se block
                    # net = squeeze_excitation_layer(net, num_channels, ratio=16, layer_name='dense_block_1')

                    net, num_channels = self.dense_block(net, num_channels)
                    print('denseblock1:', net.shape)

                with tf.variable_scope('transition_1'):
                    # add se block
                    # net = squeeze_excitation_layer(net, num_channels, ratio=16, layer_name='transition_1')

                    # feature map size: 32*256 -> 16*128
                    net, num_channels = self.transition_layer(net, num_channels)

                with tf.variable_scope('dense_block_2'):
                    # add se block
                    # net = squeeze_excitation_layer(net, num_channels, ratio=16, layer_name='dense_block_2')

                    net, num_channels = self.dense_block(net, num_channels)
                    print('denseblock2:', net.shape)

                with tf.variable_scope('transition_2'):
                    # add se block
                    # net = squeeze_excitation_layer(net, num_channels, ratio=16, layer_name='transition_2')

                    # feature map size: 16*128 -> 8*64
                    # 将transition的nchannels缩小为128
                    # num_channels = 128
                    net, num_channels = self.transition_layer(net, num_channels)

                with tf.variable_scope('dense_block_3'):
                    # add se block
                    # net = squeeze_excitation_layer(net, num_channels, ratio=16, layer_name='dense_block_3')

                    net, num_channels = self.dense_block(net, num_channels)
                    print('denseblock3:', net.shape)
                    
                with tf.variable_scope('transition_3'):
                    # add se block
                    # net = squeeze_excitation_layer(net, num_channels, ratio=16, layer_name='transition_3')

                    # feature map size: 8*64 -> 4*64
                    net, num_channels = self.transition_layer(net, num_channels, pool_stride=[2, 1])
                    # print('output:', net.shape)

                # with tf.variable_scope('global_average_pooling'):
                #     # net = slim.fully_connected(net, num_channels)
                #     net = slim.avg_pool2d(net, kernel_size=[8, 2])

                # with tf.variable_scope('transition_3'):
                #     # feature map size: 8*64 -> 4*64
                #     net, num_channels = self.transition_layer(net, num_channels, pool_stride=[2, 2],
                #                                               compression_factor=1)
                #
                # with tf.variable_scope('transition_4'):
                #     # feature map size: 4*64 -> 2*64
                #     net, num_channels = self.transition_layer(net, num_channels, pool_stride=[2, 1],
                #                                               compression_factor=1)
                #
                # with tf.variable_scope('transition_5'):
                #     # feature map size: 2*64 -> 1*64
                #     net, num_channels = self.transition_layer(net, num_channels, pool_stride=[2, 1],
                #                                               compression_factor=1)

                self.end_points = utils.convert_collection_to_dict(end_points_collection)
                self.net = net
                print('output:', net.shape)


    def dense_block(self, inputs, num_channels):
        net = slim.repeat(inputs, self.params.layers_per_block, self.block_inner_layer)
        # print('dense_block: ', net.shape)
        num_channels += self.params.growth_rate * self.params.layers_per_block
        print('num_chn:', num_channels)
        # for dwise_conv
        # num_channels = self.params.growth_rate * self.params.layers_per_block

        return net, num_channels

    def transition_layer(self, inputs, num_filter, compression_factor=0.5, pool_stride=[2, 2]):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        num_filter = int(compression_factor * num_filter)
        output = self.composite_function(inputs, num_filter, kernel_size=[1, 1])
        output = self.dropout(output)
        output = slim.avg_pool2d(output, [2, 2], stride=pool_stride)
        return output, num_filter

    def block_inner_layer(self, inputs, scope="block_inner_layer"):
        with tf.variable_scope(scope):
            if self.params.bc_mode:
                # print(inputs.shape)
                bottleneck_out = self.bottleneck(inputs)
                # print(bottleneck_out.shape)
                _output = self.composite_function(bottleneck_out, self.params.growth_rate)
            else:
                _output = self.composite_function(inputs, self.params.growth_rate)

            output = tf.concat(axis=3, values=(inputs, _output))
            # print('output: ', output.shape)
            return output

    def bottleneck(self, inputs):
        with tf.variable_scope("bottleneck"):
            num_channels = self.params.growth_rate * 4
            output = slim.batch_norm(inputs)
            # print('bottleneck_bn: ', output.shape)
            output = tf.nn.relu(output)
            output = slim.conv2d(output, num_channels, [1, 1], padding='VALID', activation_fn=None)
            print(output.shape)

            output = self.dropout(output)
        return output

    def dropout(self, inputs):
        return slim.dropout(inputs, self.params.dropout_keep_prob, is_training=self.is_training)

    def composite_function(self, inputs, num_channels, kernel_size=[1, 1]):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            output = slim.batch_norm(inputs)
            output = tf.nn.relu(output)
            output = slim.conv2d(output, num_channels, kernel_size, activation_fn=None)
            output = slim.batch_norm(output)
            output = tf.nn.relu(output)
            output = dwise_conv(output, k_h=3, k_w=3, padding='SAME')
            output = self.dropout(output)
        return output

    def arg_scope(self, is_training=True,
                  weight_decay=0.0001,
                  batch_norm_decay=0.997,
                  batch_norm_epsilon=1e-5,
                  batch_norm_scale=True):
        batch_norm_params = {
            'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS
        }

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '../../libs')
    # from tf_utils import print_endpoints

    inputs = tf.placeholder(tf.float32, [None, 32, None, 1], name="inputs")

    is_training = tf.placeholder(tf.bool, name="is_training")
    img_file = '/data01/wuyang.zhang/dl/OCR_TF_CRNN_CTC/test_data/demo.png'

    dense_net = DenseNet(inputs, is_training)
    # print(dense_net)
    # print_endpoints(dense_net, inputs, is_training, img_file)
