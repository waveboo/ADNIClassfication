# -*- coding:utf8 -*-
from keras.models import load_model
from ResNetBuild import *
from keras.layers.merge import(
    add,
    concatenate
    ) 
from keras import layers
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, GlobalAveragePooling3D


# To judge the backend is 'channels_first' or 'channels_last'
def judge_channels():
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


def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    # Bottleneck layers
    x = BatchNormalization(axis=4)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv3D(bn_size*nb_filter, (1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    # Composite function
    x = BatchNormalization(axis=4)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv3D(nb_filter, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    if drop_rate: x = Dropout(drop_rate)(x)
    return x
 
def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=4)
    return x
    
def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):
    nb_filter = int(x.shape.as_list()[-1]*compression)
    x = BatchNormalization(axis=4)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv3D(nb_filter, (1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    if is_max != 0: x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)
    else: x = AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)
    return x


class Model2Steam(object):
    @staticmethod
    def buildresnetimg(input, block_fn, repetitions, reg_factor, ispreload=False):
        judge_channels()
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
        return pool2


    @staticmethod
    def buildlenetimg(input, reg_factor):
        conv1 = Activation("relu")(Conv3D(filters=6, kernel_size=(5, 5, 5), strides=(1, 1, 1),
                                          kernel_initializer="he_normal", padding="valid",
                                          kernel_regularizer=l2(reg_factor))(input))
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv1)
        conv2 = Activation("relu")(Conv3D(filters=16, kernel_size=(5, 5, 5), strides=(1, 1, 1),
                                          kernel_initializer="he_normal", padding="valid",
                                          kernel_regularizer=l2(reg_factor))(pool1))
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv2)
        return pool2


    @staticmethod
    def builddensenetimg(inpt, reg_factor):
        growth_rate = 5
        x = Conv3D(growth_rate*2, (3, 3, 3), strides=(1,1,1), padding='same')(inpt)
        x = BatchNormalization(axis=4)(x)
        x = LeakyReLU(alpha=0.1)(x)
 
        x = DenseBlock(x, 5, growth_rate, drop_rate=0.2)
        x = TransitionLayer(x)
        x = DenseBlock(x, 5, growth_rate, drop_rate=0.2)

        x = BatchNormalization(axis=4)(x)
        x = GlobalAveragePooling3D()(x)

        return x



    @staticmethod
    def buildresnet(input1_shape, input2_shape, num_outputs, block_fn, reg_factor):
        judge_channels()
        if len(input1_shape) != 4 or len(input2_shape) != 4:
            raise ValueError("Input shape should be a tuple (conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or (channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        smri_input = Input(shape=input1_shape)
        fmri_input = Input(shape=input2_shape)

        model_smri_pool2 = Model2Steam.buildresnetimg(smri_input, block_fn, [2, 2, 2, 2], reg_factor, True)
        model_fmri_pool2 = Model2Steam.buildresnetimg(fmri_input, block_fn, [2, 2, 2, 2], reg_factor)
        added = add([model_smri_pool2, model_fmri_pool2])

        flatten1 = Flatten()(added)
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

        model = Model(inputs=[smri_input, fmri_input], outputs=dense)
        return model


    @staticmethod
    def buildlenet(input1_shape, input2_shape, num_outputs, reg_factor):
        judge_channels()
        if len(input1_shape) != 4 or len(input2_shape) != 4:
            raise ValueError("Input shape should be a tuple (conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or (channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        smri_input = Input(shape=input1_shape)
        fmri_input = Input(shape=input2_shape)

        model_smri_pool2 = Model2Steam.buildlenetimg(smri_input, reg_factor)
        model_fmri_pool2 = Model2Steam.buildlenetimg(fmri_input, reg_factor)
        added = add([model_smri_pool2, model_fmri_pool2])

        flatten1 = Flatten()(added)
        fc1 = Dense(units=120, kernel_initializer="he_normal", activation="relu",
                    kernel_regularizer=l2(reg_factor))(flatten1)
        fc2 = Dense(units=84, kernel_initializer="he_normal", activation="relu",
                    kernel_regularizer=l2(reg_factor))(fc1)
        if num_outputs > 1:
            fc3 = Dense(units=num_outputs, kernel_initializer="he_normal", activation="softmax",
                        kernel_regularizer=l2(reg_factor))(fc2)
        else:
            fc3 = Dense(units=num_outputs, kernel_initializer="he_normal", activation="sigmoid",
                        kernel_regularizer=l2(reg_factor))(fc2)
        model = Model(inputs=[smri_input, fmri_input], outputs=fc3)
        return model


    @staticmethod
    def builddensenet(input1_shape, input2_shape, num_outputs, reg_factor):
        judge_channels()
        if len(input1_shape) != 4 or len(input2_shape) != 4:
            raise ValueError("Input shape should be a tuple (conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or (channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        smri_input = Input(shape=input1_shape)
        fmri_input = Input(shape=input2_shape)

        smri_dense1 = Model2Steam.builddensenetimg(smri_input, reg_factor)
        fmri_dense1 = Model2Steam.builddensenetimg(fmri_input, reg_factor)
        added = add([smri_dense1, fmri_dense1])

        
        if num_outputs > 1:
            fc2 = Dense(units=num_outputs, kernel_initializer="he_normal", activation="softmax",
                        kernel_regularizer=l2(reg_factor))(added)
        else:
            fc2 = Dense(units=num_outputs, kernel_initializer="he_normal", activation="sigmoid",
                        kernel_regularizer=l2(reg_factor))(added)
        model = Model(inputs=[smri_input, fmri_input], outputs=fc2)
        return model


    @staticmethod
    def build_model_resnet(input1_shape, input2_shape, num_outputs, reg_factor=1e-4):
        return Model2Steam.buildresnet(input1_shape, input2_shape, num_outputs, basic_block3d, reg_factor=reg_factor)

    @staticmethod
    def build_model_lenet(input1_shape, input2_shape, num_outputs, reg_factor=1e-4):
        return Model2Steam.buildlenet(input1_shape, input2_shape, num_outputs, reg_factor=reg_factor)

    @staticmethod
    def build_model_densenet(input1_shape, input2_shape, num_outputs, reg_factor=1e-4):
        return Model2Steam.builddensenet(input1_shape, input2_shape, num_outputs, reg_factor=reg_factor)


