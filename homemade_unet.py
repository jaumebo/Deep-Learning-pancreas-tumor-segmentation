import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, \
                                    MaxPooling3D, UpSampling3D, \
                                    Activation, BatchNormalization, PReLU, \
                                    Conv3DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model


def convolution_block(input_layer,
                      filters,
                      name,
                      kernel_size = 3,
                      strides = (1,1,1),
                      padding = 'same',
                      batch_normalization = True,
                      activation = 'relu'):

    output_layer = Conv3D(filters=filters,
                          name=name,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          data_format='channels_first')(input_layer)
    
    if batch_normalization:
        output_layer = BatchNormalization(axis=1)(output_layer)
    
    if activation in ['relu','sigmoid']:
        return Activation(activation)(output_layer)
    elif activation == 'softmax':
        return tf.nn.softmax(output_layer, axis=1, name='output_softmax')
    else:
        return Activation('relu')(output_layer)


def unet_model_3d_graph(input_shape,n_labels,pool_size=2,initial_learning_rate=0.00001,kernel_up_convolution=(2,2,2)):

    inputs = Input(input_shape)

    #First level
    x = convolution_block(inputs,filters=64,name='lvl1_Conv3D1')
    x = convolution_block(x,filters=64,name='lvl1_Conv3D2')
    connection_1 = x
    x = MaxPooling3D(pool_size=pool_size,name='lvl1_MaxPooling',data_format='channels_first')(x)

    #Second level
    x = convolution_block(x,filters=128,name='lvl2_Conv3D1')
    x = convolution_block(x,filters=128,name='lvl2_Conv3D2')
    connection_2 = x
    x = MaxPooling3D(pool_size=pool_size,name='lvl2_MaxPooling',data_format='channels_first')(x)

    #Third level
    x = convolution_block(x,filters=256,name='lvl3_Conv3D1')
    x = convolution_block(x,filters=256,name='lvl3_Conv3D2')
    connection_3 = x
    x = MaxPooling3D(pool_size=pool_size,name='lvl3_MaxPooling',data_format='channels_first')(x)

    #Fourth level
    x = convolution_block(x,filters=512,name='lvl4_Conv3D1')
    x = convolution_block(x,filters=512,name='lvl4_Conv3D2')
    connection_4 = x
    x = MaxPooling3D(pool_size=pool_size,name='lvl4_MaxPooling',data_format='channels_first')(x)

    #Base level
    x = convolution_block(x,filters=1024,name='base_Conv3D1')
    x = convolution_block(x,filters=1024,name='base_Conv3D2')
    x = Conv3DTranspose(filters=512, kernel_size=(kernel_up_convolution), strides=(2,2,2), 
                        name='base_UpConv', data_format='channels_first')(x)

    #Fourth up-level
    x = concatenate([x,connection_4],axis=1)
    x = convolution_block(x,filters=512,name='up-lvl4_Conv3D1')
    x = convolution_block(x,filters=512,name='up-lvl4_Conv3D2')
    x = Conv3DTranspose(filters=256, kernel_size=kernel_up_convolution, strides=(2,2,2), 
                        name='lvl4_UpConv', data_format='channels_first')(x)

    #Third up-level
    x = concatenate([x,connection_3],axis=1)
    x = convolution_block(x,filters=256,name='up-lvl3_Conv3D1')
    x = convolution_block(x,filters=256,name='up-lvl3_Conv3D2')
    x = Conv3DTranspose(filters=128, kernel_size=kernel_up_convolution, strides=(2,2,2), 
                        name='lvl3_UpConv', data_format='channels_first')(x)

    #Second up-level
    x = concatenate([x,connection_2],axis=1)
    x = convolution_block(x,filters=128,name='up-lvl2_Conv3D1')
    x = convolution_block(x,filters=128,name='up-lvl2_Conv3D2')
    x = Conv3DTranspose(filters=64, kernel_size=kernel_up_convolution, strides=(2,2,2), 
                        name='lvl2_UpConv', data_format='channels_first')(x)

    #First up-level
    x = concatenate([x,connection_1],axis=1)
    x = convolution_block(x,filters=64,name='up-lvl1_Conv3D1')
    x = convolution_block(x,filters=64,name='up-lvl1_Conv3D2')

    #Output
    outputs = convolution_block(x,
                                filters=n_labels,
                                name='output_layer',
                                kernel_size=1,
                                batch_normalization=False,
                                activation='softmax')

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss='binary_crossentropy')
        
    #print(model.summary()) 
    
    return model

def unet_model_3d(input_shape,n_labels,loss='binary_crossentropy',pool_size=2,initial_learning_rate=0.00001,kernel_up_convolution=(2,2,2),gpus=1):
    
    if gpus==1:
        
        model = unet_model_3d_graph(input_shape,
                                    n_labels,
                                    pool_size,
                                    initial_learning_rate,
                                    kernel_up_convolution=(2,2,2))
        
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss)
        
        return model
    
    elif gpus>1:
            
        with tf.device('/cpu:0'):
            model = unet_model_3d_graph(input_shape,
                                    n_labels,
                                    pool_size,
                                    initial_learning_rate,
                                    kernel_up_convolution=(2,2,2))
        
        parallel_model = multi_gpu_model(model,gpus=gpus)
        parallel_model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss)
        
        return model
        
        

