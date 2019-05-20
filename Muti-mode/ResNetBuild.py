# -*- coding:utf8 -*-
import math
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


global DIM1_AXIS
global DIM2_AXIS
global DIM3_AXIS
global CHANNEL_AXIS
if K.image_data_format() == 'channels_last':
    DIM1_AXIS = 1
    DIM2_AXIS = 2
    DIM3_AXIS = 3
    CHANNEL_AXIS = 4
else:
    CHANNEL_AXIS = 1
    DIM1_AXIS = 2
    DIM2_AXIS = 3
    DIM3_AXIS = 4


# Build a BatchNormalization to ReLU connection
def bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


# Build a conv3d to BatchNormalization to ReLU connection, this will be used at start
def conv3d_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    # f is the nest function control the input
    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(input)
        return bn_relu(conv)
    return f


# Build a BatchNormalization to ReLU to conv3d connection, this is the most common(suggest) connection in this network
def bn_relu_conv3d(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        activation = bn_relu(input)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f


# Build a shortcut to combine the input and residual
def shortcut3d(input, residual):
    # math.ceil to prevent layer size not match
    stride_dim1 = math.ceil(input._keras_shape[DIM1_AXIS] / residual._keras_shape[DIM1_AXIS])
    stride_dim2 = math.ceil(input._keras_shape[DIM2_AXIS] / residual._keras_shape[DIM2_AXIS])
    stride_dim3 = math.ceil(input._keras_shape[DIM3_AXIS] / residual._keras_shape[DIM3_AXIS])
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    # If the input's shape is not same with residual's, conv the input to the same size with residual
    if (stride_dim1 > 1) or (stride_dim2 > 1) or (stride_dim3 > 1) or (not equal_channels):
        shortcut = Conv3D(
            filters=residual._keras_shape[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal",
            padding="valid",
            kernel_regularizer=l2(1e-4)
            )(input)
    return add([shortcut, residual])


# Build a 3D residual block with the block function
def residual_block3d(block_function, filters, kernel_regularizer, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            input = block_function(filters=filters, strides=strides,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0)
                                   )(input)
        return input
    return f


# Block function 1, basic block function with (3*3) to (3*3)
def basic_block3d(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=strides, padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=kernel_regularizer
                           )(input)
        else:
            conv1 = bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   strides=strides,
                                   kernel_regularizer=kernel_regularizer
                                   )(input)

        residual = bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                  kernel_regularizer=kernel_regularizer
                                  )(conv1)
        return shortcut3d(input, residual)
    return f


# Block function 2, bottleneck block function with (1*1) to (3*3) to (1*1)
def bottleneck_block3d(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                              strides=strides, padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=kernel_regularizer
                              )(input)
        else:
            conv_1_1 = bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
                                      strides=strides,
                                      kernel_regularizer=kernel_regularizer
                                      )(input)

        conv_3_3 = bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                  kernel_regularizer=kernel_regularizer
                                  )(conv_1_1)
        residual = bn_relu_conv3d(filters=filters * 4, kernel_size=(1, 1, 1),
                                  kernel_regularizer=kernel_regularizer
                                  )(conv_3_3)

        return shortcut3d(input, residual)
    return f


class Resnet3DBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, reg_factor):
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple (conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or (channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")
        input = Input(shape=input_shape)

        conv1 = conv3d_bn_relu(filters=64, kernel_size=(7, 7, 7),
                               strides=(2, 2, 2),
                               kernel_regularizer=l2(reg_factor)
                               )(input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                             padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = residual_block3d(block_fn, filters=filters,
                                     kernel_regularizer=l2(reg_factor),
                                     repetitions=r, is_first_layer=(i == 0)
                                     )(block)
            filters *= 2

        block_output = bn_relu(block)
        pool2 = AveragePooling3D(pool_size=(block._keras_shape[DIM1_AXIS],
                                            block._keras_shape[DIM2_AXIS],
                                            block._keras_shape[DIM3_AXIS]),
                                 strides=(1, 1, 1))(block_output)

        flatten1 = Flatten()(pool2)
        # flatten1 = Dropout(rate=0.4)(Flatten()(pool2))
        if num_outputs > 1:
            dense = Dense(units=num_outputs,
                          kernel_initializer="he_normal",
                          activation="softmax",
                          kernel_regularizer=l2(reg_factor))(flatten1)
        else:
            dense = Dense(units=num_outputs,
                          kernel_initializer="he_normal",
                          activation="sigmoid",
                          kernel_regularizer=l2(reg_factor))(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, reg_factor=1e-4):
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block3d, [2, 2, 2, 2], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, reg_factor=1e-4):
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block3d, [3, 4, 6, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, reg_factor=1e-4):
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck_block3d, [3, 4, 6, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, reg_factor=1e-4):
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck_block3d, [3, 4, 23, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, reg_factor=1e-4):
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck_block3d, [3, 8, 36, 3], reg_factor=reg_factor)


if __name__ == '__main__':
    Resnet3DBuilder.build_resnet_18((64, 64, 32, 1), 1)

