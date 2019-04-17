import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


class DenseNet(Model):

    def __init__(self, 
                 block_config=(6, 12, 24, 16), init_features=64,
                 growth_rate=32, drop_rate=0, bottle_neck=4, compression=0.5):
        super(DenseNet, self).__init__()

        # Network Parameters
        self.block_config=block_config
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.bottle_neck = bottle_neck
        self.compression = compression

        # Network Components
        self.conv1 = layers.Conv2D(
            filters=init_features,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='same',
            activation='relu')
        self.max_pool1 = layers.MaxPooling2D(
            pool_size=(3, 3), 
            strides=(2, 2), 
            padding='same')
        self.blocks = [([self._dense_layer] * n) for n in block_config]
        self.transitions = [self._transition_layer] * len(block_config)
        self.avg_pool1 = layers.AveragePooling2D()
        self.dense1 = layers.Dense(units=10)

    def _dense_layer(self):
        bn1 = layers.BatchNormalization()
        conv1 = layers.Conv2D(
            filters=self.bottle_neck * self.growth_rate, 
            kernel_size=(1, 1), strides=(1, 1), 
            padding='same', 
            activation='relu')
        bn2 = layers.BatchNormalization()
        conv2 = layers.Conv2D(
            filters=self.growth_rate, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='same', 
            activation='relu')
        return tf.keras.Sequential([bn1, conv1, bn2, conv2])

    def _transition_layer(self, in_feats):
        bn1 = layers.BatchNormalization()
        conv1 = layers.Conv2D(
            filters=int(in_feats * self.compression),
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu')
        avg_pool1 = layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid')

        return tf.keras.Sequential([bn1, conv1, avg_pool1])

    def call(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        for i in range(len(self.block_config)):
            x_list = [x]
            block = self.blocks[i]
            for layer in block:
                x = layer()(tf.concat(x_list, -1))
                x_list.append(x)

            x = self.transitions[i](x.shape.as_list()[-1])(x)
            
        x = layers.AveragePooling2D(
                pool_size=(x.shape.as_list()[1], x.shape.as_list()[2]), 
                strides=(1, 1))(x)
                
        return self.dense1(x)


            
