'''
PolyNet.

This model is ResNet network augmented the Keras 
implementation of the PolyNet module.  Basically, PolyNets 
serve to increase the structural complexity of ResNet 
networks.  Because ResNets reach a saturation point for 
accuracy gains by either adding deep or width to the 
network, PolyNet are a solution for further improving 
accuracy in those cases.  PolyNets essentially offer 
residual connections the ability to re-use weights (in a 
pseudo-recurrent fashion), perform two distinct weight 
operations, and cascade weight operations to integrate 
variation into the model.  It has been shown that these 
PolyNet connections though only seem to be really beneficial 
when the network is very deep. 

# Reference:
    https://arxiv.org/pdf/1611.05725.pdf
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings, math

import numpy as np
import tensorflow as tf

import keras
from keras.models import Model
from keras.datasets import cifar100

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint

import keras.backend as K
from keras.optimizers import *
from keras import metrics

from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape


DEFAULT_FINAL_DROPOUT_RATE=0.1
DEFAULT_BLOCK_SIZES=[8, 15, 6]
DEFAULT_INIT_FILTER=128

MPOLYN="mpoly-N"
POLYN="poly-N"
NWAY="N-way"



def PolyNet(input_shape=None, nb_layers_per_block=DEFAULT_BLOCK_SIZES, 
                init_nb_filters=DEFAULT_INIT_FILTER, final_dropout=0.0, 
                weight_decay=1E-4, include_top=True, input_tensor=None, classes=100, 
                activation='softmax'):
			 
    '''Instantiate the PolyNet architecture. 
    
        The dimension ordering convention used by the model is the one specified in your Keras 
        config file.
        
        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            nb_layers_per_block: number of layers in each block of residual connections
                Must be a list of residual connections between transition layers.
            init_nb_filters: number of channels in layer after first convolutional operation
            final_dropout: dropout rate of final FC layer
            weight_decay: weight decay factor
            include_top: whether to include the fully-connected
                layer at the top of the network.
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
                
        # Returns
            A Keras model instance.
        '''

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

        
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_poly_net(classes, img_input, include_top, nb_layers_per_block, 
                            init_nb_filters, final_dropout, weight_decay, activation)

                            
    # Ensure that the model takes into account any predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    
    # Create model
    model = Model(inputs, x, name='densenet')

    
    # Return final model
    return model


def __regular_residual_op(x, num_filters, weight_decay=1E-4):
    ''' Apply Relu, 1x1 Conv2D, BN, Relu, 3x3 ConvSeparable2D, BN for residual output
    Args:
        ip: keras input tensor
        num_filters: number of filters
        weight_decay: weight decay factor
        
    Returns: keras output tensor (representing only residual term)
    '''
    
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Activation('relu')(x)        
    x = Conv2D(int(num_filters // 2), (1, 1), kernel_initializer='he_uniform', padding='same',
               use_bias=True, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    x = Activation('relu')(x)        
    x = SeparableConv2D(num_filters, (3, 3), kernel_initializer='he_uniform', padding='same',
               use_bias=True, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

                           
    return x
    
    
def __transition_block(x, nb_filter, weight_decay=1E-4):
    ''' Apply Relu, 1x1 Conv2D, BN, and then Maxpooling2D (to reduce spatial dimensions)
    Args:
        x: keras input tensor
        nb_filter: number of filters for output tensor
        weight_decay: weight decay factor
        
    Returns: keras output tensor 
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Activation('relu')(x)        
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_uniform', padding='same',   
               use_bias=True, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
						   beta_regularizer=l2(weight_decay))(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
                           
    return x
    
    
# A simple "poly" model for use in a "poly"-module (in 'do_poly_operation' function)
def make_poly_function(input, layer, weight_decay=1E-4):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    num_filters = K.int_shape(input)[concat_axis]
    tmp_input = Input(shape=K.int_shape(input)[1:])
    
    output = Activation('relu')(tmp_input)
    output = Conv2D(int(num_filters // 2), (1, 1), kernel_initializer='he_uniform', padding='same',
                        use_bias=True, kernel_regularizer=l2(weight_decay))(output)
    output = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
						   beta_regularizer=l2(weight_decay))(output)
    
    output = Activation('relu')(output)
    output = layer(output)
    output = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
						   beta_regularizer=l2(weight_decay))(output)
    
    
    # Return model
    return Model(inputs=tmp_input, outputs=output)
    

# Perform a specific "poly" operations given a set of layers 
def do_poly_operation(input, layers_ops, type, N, weight_decay=1E-4):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    # Get necessary poly functions for a poly-module
    poly_funcs = []
    for layer_op in layers_ops:
        poly_funcs.append(make_poly_function(input, layer_op, weight_decay))

        
    # Get initial output
    output = poly_funcs[0](input)
    
    # Add remaining parts of "mpoly-N" module
    if type == MPOLYN:
        assert len(poly_funcs) == N, "Incorrect number of poly functions provided."
                    
        y = output
        for f in poly_funcs[1:]:
            y = f(y)
            output = add([output, y])
    
    # Add remaining parts of "poly-N" module
    elif type == POLYN:
        assert len(poly_funcs) == 1, "Incorrect number of poly functions provided."
        
        y = output
        for _ in range(N-1):
            y = poly_funcs[0](y)
            output = add([output, y])

    # Add remaining parts of "N-way" module
    elif type == NWAY:
        assert len(poly_funcs) == N, "Incorrect number of poly functions provided."
        
        for f in poly_funcs[1:]:
            y = f(input)
            output = add([output, y])
    
    else:
        assert False, "Invalid flag passed to 'do_poly_operation' function."
    
    
    # Return resulting residual term output logits
    return output

    
# Do a user-specified sequence of poly_operate calls (on a given input)
def __poly_inception_block(x, num_sequences, poly_tuples, weight_decay=1E-4):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    # Get each poly-residual 'num_sequences' times and add to main feature channels
    num_filters = K.int_shape(x)[concat_axis]
    for _ in range(num_sequences):
        for type, N_value, layer_funcs, filter_sizes in poly_tuples:
            layer_ops = [ ]
            if type == MPOLYN or type == NWAY:
                for i in range(N_value):
                    layer_ops.append(layer_funcs[i](num_filters, filter_sizes[i], 
                                                    kernel_initializer='he_uniform', 
                                                    padding='same', use_bias=True, 
                                                    kernel_regularizer=l2(weight_decay)))
            else:
                layer_ops.append(layer_funcs[0](num_filters, filter_sizes[0], 
                                                kernel_initializer='he_uniform', 
                                                padding='same', use_bias=True, 
                                                kernel_regularizer=l2(weight_decay)))
            
            y = do_poly_operation(x, layer_ops, type, N_value, weight_decay)
            x = add([x, y])

            
    # Return final output logits (after block has been completed)    
    return x

    
def __create_poly_net(nb_classes, img_input, include_top, nb_layers_per_block, 
                        init_nb_filters, final_dropout, weight_decay=1E-4, 
                        activation='softmax'):
    
    ''' Build a ResNet model with "poly" modules for the middle blocks
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_layers_per_block: number of layers in each block of residual connections
            Must be a list of residual connections between transition layers.
        init_nb_filters: number of channels in layer after first convolutional operation
        final_dropout: dropout rate of final FC layer
        weight_decay: weight decay
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
                
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1


    # Initial convolution
    nb_filter = init_nb_filters
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_uniform', padding='same', name='initial_conv2D',
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
						   beta_regularizer=l2(weight_decay))(x)
	
    
    # Do a bunch of Res-Nets ops
    num_layers = nb_layers_per_block[0]
    for _ in range(num_layers):
        y = __regular_residual_op(x, nb_filter, weight_decay)        
        x = add([x, y])
        
        
    # Do 1st transition (with 2.5X channel increase)
    nb_filter = int(nb_filter * 2.5)
    x = __transition_block(x, nb_filter)    
                    
                    
	# Do mixed PolyNet sequence
    for num_layer in nb_layers_per_block[1:-1]:
        # PolyNet logic
        if False:
            poly_tuples = []
            poly_tuples.append((MPOLYN, 3, [SeparableConv2D] * 3, [(3, 3)] * 3))
            poly_tuples.append((POLYN, 3, [SeparableConv2D], [(3, 3)]))
            poly_tuples.append((NWAY, 2, [SeparableConv2D] * 2, [(3, 3), (5, 5)]))
            poly_tuples.append((MPOLYN, 3, [SeparableConv2D] * 3, [(3, 3)] * 3))
            poly_tuples.append((POLYN, 3, [SeparableConv2D], [(3, 3)]))

            num_el_in_seq = len(poly_tuples)
            assert num_layer % num_el_in_seq == 0, \
                        ("Middle blocks should be have number of layers divisible by %d" % num_el_in_seq)
            num_sequences = int(num_layer // num_el_in_seq)

            x = __poly_inception_block(x, num_sequences, poly_tuples, weight_decay)
            
        else:
            for _ in range(num_layer * 2):
                y = __regular_residual_op(x, nb_filter, weight_decay)
                x = add([x, y])            

        # Do middle transitions			
        nb_filter = int(nb_filter * 2)
        x = __transition_block(x, nb_filter)                

    # Do final ResNet connections    
    num_layers = 6
    for _ in range(num_layers):
        y = __regular_residual_op(x, nb_filter, weight_decay)
        x = add([x, y])
        
        
    # Once done, take an average pool of the spatial dimensions        
    x = GlobalAveragePooling2D(name="final_embeddings")(x)

    
    # Add final FC layer if requested
    if include_top:
        x = Activation('relu')(x)
        if final_dropout:
            x = Dropout(final_dropout, name="final_dropout")(x)
            	
        x = Dense(nb_classes, activation=activation, kernel_regularizer=l2(weight_decay), 
                        bias_regularizer=l2(weight_decay))(x)

                        
    # Return final embeddings logits
    return x