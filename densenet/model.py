import numpy as np
import tensorflow as tf


def dense_layer(input_features, training, growth_rate, dropout_rate,
                bottle_neck_size=4):
    """layer within the DenseBlock

    Args:
        input_features (Tensor): 4 dimensional input tensor (B, H, W, C)
        training (bool): flag for training mode
        growth_rate (int): hyper parameter that effects the number of filters
                           in each block
        dropout_rate (float): rate to dropout channels after each block
        bottle_neck_size (int): multiplicative factor of the growth rate which
                                effects the output features of the bottleneck
                                convolution of the layer. Defaults to 4 as per
                                the paper.

    Returns:
        output feature Tensor

    """

    net = tf.layers.batch_normalization(input_features, training=training)
    net = tf.layers.conv2d(
        net,
        filters=bottle_neck_size * growth_rate,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='SAME',
        activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.conv2d(
        net,
        filters=growth_rate,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        activation=tf.nn.relu)

    return net


def transition_layer(input_features, training, compression=0.5):
    """layer between DenseBlocks

    Args:
        input_features (Tensor): 4 dimensional input tensor (B, H, W, C)
        training (bool): flag for training mode
        compression (float): factor in which the bottleneck layer 'compresses'
                             the number of features of each DenseBlock.
                             Defaults to 0.5 as per the paper.

    Returns:
        output feature Tensor

    """
    net = tf.layers.batch_normalization(input_features, training=training)
    net = tf.layers.conv2d(
        net,
        filters=int(input_features.shape[-1].value * compression),
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='SAME',
        activation=tf.nn.relu)
    net = tf.layers.average_pooling2d(
        net,
        pool_size=(2, 2),
        strides=(2, 2),
        padding='VALID')
    return net


def densenet(input_features, training,
             dropout_rate=0, growth_rate=12, init_features=64, num_classes=2,
             block_config=(6, 12, 24, 16)):
    """DenseNet from the paper "Densely Connected Convolutional Networks"

    Default arguments are set for binary classification with Densenet121

    Args:
        input_features (Tensor): 4 dimensional input tensor (B, H, W, C)
        training (bool): flag for training mode
        dropout_rate (float): rate to dropout channels after each block
        growth_rate (int): hyper parameter that effects the number of filters
                           in each block
        init_features (int): number of filters to output for initial convolution
        num_classes (int): number of logits that the network will output
        block_config (tuple): tuple of ints that determines the amount of
                              DenseBlocks as well as their depth

    Returns:
        logits of shape (B, num_classes)

    """

    with tf.name_scope('Densenet'):
        net = tf.layers.conv2d(
            input_features,
            filters=init_features,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='SAME',
            activation=tf.nn.relu,
            name='init_conv')
        net = tf.layers.max_pooling2d(
            net,
            pool_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            name='init_pool')

        for i in range(len(block_config)):
            with tf.name_scope('DenseBlock{}'.format(i)):
                net_list = [net]
                for j in range(block_config[i]):
                    net = dense_layer(tf.concat(net_list, axis=-1),
                                      training, growth_rate, dropout_rate)
                    net = tf.layers.dropout(
                        net, rate=dropout_rate, training=training)
                    net_list.append(net)
            if i != len(block_config) - 1:
                net = transition_layer(net, training)

    with tf.name_scope('Classifier'):
        net = tf.layers.average_pooling2d(
            net,
            pool_size=(net.shape[1].value, net.shape[2].value),
            strides=(1, 1),
            name='global_average_pool')
        net = tf.layers.dense(net, num_classes)

    return net


def main():
    batch_size, height, width, channels = [32, 256, 512, 1]

    input_pl = tf.placeholder(dtype=tf.float32,
                              shape=[batch_size, height, width, channels])

    training_pl = tf.placeholder(dtype=tf.bool)
    logits = densenet(input_pl, training_pl)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(logits,
                       feed_dict={
                           input_pl: np.random.rand(
                               batch_size, height, width, channels),
                           training_pl: True
                       })
        print(out)


if __name__ == '__main__':
    main()
