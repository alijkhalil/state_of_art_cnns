'''
Dual Path Networks for Keras.

In short, this network combines the idea of a ResNet with the recent
advent of the DenseNet.  In theory, it should provide the benefits of 
both models for a more efficient and expressive solution.

# Reference:
    https://arxiv.org/pdf/1707.01629.pdf
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

import math
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

from keras.layers.core import Dense, Lambda, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape


USE_DROPPATH=True
DEFAULT_DEATH_RATE=0.15
DEFAULT_DROPOUT_RATE=0.15
DEFAULT_WEIGHT_DECAY=1E-4

DEFAULT_NUM_EPOCHS=100
DEFAULT_DEPTH=58
DEFAULT_NUM_BLOCKS=3
DEFAULT_INIT_FILTERS=128
DEFAULT_LAYERS_PER_BLOCK=[12, 32, 10]
DEFAULT_NUM_CLASSES=100


def DualPathNetwork(input_shape=None, depth=DEFAULT_DEPTH, nb_dense_block=DEFAULT_NUM_BLOCKS, 
                init_filters=DEFAULT_INIT_FILTERS, nb_layers_per_block=DEFAULT_LAYERS_PER_BLOCK, 
                use_droppath=USE_DROPPATH, dropout_rate=DEFAULT_DROPOUT_RATE, 
                weight_decay=DEFAULT_WEIGHT_DECAY, include_top=True, input_tensor=None, 
                classes=DEFAULT_NUM_CLASSES, activation='softmax'):
                
    '''Instantiate a slight modification of the Dual Path Network architecture.
        The main tenants are the same, but some liberities were taken to combine 
        it with other prominent work in computer vision.  As such, multiple filter 
        sizes are used for the "grouped" or separable convolutions.  Also, an 
        optional dropout is added to the final layer.  Aside from that, it is very 
        similar to the orginal construct.

        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            depth: number or layers in the DPN
            nb_dense_block: number of DPN blocks as basic building blocks of the model (generally 3 to 5)
            growth_rate: number of filters to add per layher block in "dense" path
            init_filters: initial number of filters
                Should be 128 unless the model is extremely small or large
            nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the network depth.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must be "nb_dense_block" in size.
            dropout_rate: dropout rate of final layer
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

    x, drop_table = __create_dual_path_net(classes, img_input, include_top, depth, 
                                            nb_dense_block, init_filters, 
                                            nb_layers_per_block, dropout_rate, use_droppath, 
                                            weight_decay, activation)

    # Ensure that the model takes into account any potential predecessors of 'input_tensor'
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create and return the model
    model = Model(inputs, x, name='dpn_net')

    
    return model, drop_table


def __conv_block(ip, nb_filter, smaller_filters=False, weight_decay=DEFAULT_WEIGHT_DECAY):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D for downsize and then two distinct separable conv ops
    Args:
        ip: input keras tensor
        nb_filter: final number of filters to output
        smaller_filters: flag for trigerring a 2x2 separable conv (instead of a 5x5 one)
        weight_decay: weight decay factor
        
    Returns: keras tensor with after block using two filter sizes
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    if smaller_filters:
        alt_filter_size = 2
    else:
        alt_filter_size = 5
        
    # Reduce channel size of the input layer, but more so, add a non-linearity and combine all features
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)    
    x = Conv2D(int(nb_filter // 3), (1, 1), kernel_initializer='he_uniform', padding='same', 
                    use_bias=True, kernel_regularizer=l2(weight_decay))(x)

    
    # In the vein of NAS convolutional cells, perform two different sized seperable convolutions and add them
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)    
    x3by3 = SeparableConv2D(nb_filter, (3, 3), padding='same', use_bias=True)(x)
    x_other = SeparableConv2D(nb_filter, (alt_filter_size, alt_filter_size), padding='same', use_bias=True)(x)

    output = add([x3by3, x_other])
    
    
    # Return output layer
    return output


def __transition_block(prev_ip, cur_ip, new_nb_filter, weight_decay=DEFAULT_WEIGHT_DECAY):
    ''' Apply BatchNorm, Relu, Conv2D to the last two transition inputs and then combine to downsize
    Args:
        prev_ip: keras tensor (e.g. input of current block)
        cur_ip: keras tensor (e.g. output of current block)
        new_nb_filter: new number of filters
        weight_decay: weight decay factor
        
    Returns: keras tensor after applying transition operations on both inputs
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if prev_ip is None or not K.is_keras_tensor(prev_ip):
        prev_ip = cur_ip
    
    # Apply operations to block input and output to reduce between any arbitary layer in the model
    x1 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(prev_ip)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(new_nb_filter, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=True,
               kernel_regularizer=l2(weight_decay))(x1)    
    x1 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(new_nb_filter, (5, 5), kernel_initializer='he_uniform', padding='same', use_bias=True,
               kernel_regularizer=l2(weight_decay))(x1)
    x1 = AveragePooling2D((2, 2), strides=(2, 2))(x1)
    
    
    x2 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(cur_ip)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(new_nb_filter, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=True,
               kernel_regularizer=l2(weight_decay))(x2)        
    x2 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x2)
    x2 = Activation('relu')(x2)                   
    x2 = SeparableConv2D(new_nb_filter, (3, 3), strides=(2, 2), padding='same', use_bias=True)(x2)
    
    # Add them together and output them
    output = add([x1, x2])
    
    
    # Return output layer
    return output

    
# Helper Layer for getting a subsection of a layer
#  
# Usage example:    x = crop(2, 5, 10)(x)
def crop(dimension, start, end=0):
    def func(x, dimension=dimension, start=start, end=end):
        if not end:
            end = K.int_shape(x)[dimension]

        if dimension < 0:
            dimension += len(K.int_shape(x))

        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
			
    return Lambda(func)


# Layer for scaling down activations at test time
def scale_activations(drop_rate):
    def func(x, drop_rate=drop_rate):
        scale = K.ones_like(x) - drop_rate
        return K.in_test_phase(scale * x, x)
    
    return Lambda(func)


# Layer for dropping path based on a "gate" variable
#   Input should be formatted: [drop_path, normal_path] 
#   "Gate" variable should be set to 1 to return "normal_path"
def drop_path(gate):
    def func(tensors, gate=gate):
        return K.switch(gate, tensors[1], tensors[0])
			
    return Lambda(func)      
    
    
# Wrapper for add combination layer (with drop path functionality incorporated)
def res_add(layers, drop_dict):
    if drop_dict is not None:
        # Get death_rate and drop gate variables from table
        gate = drop_dict["gate"]
        death_rate = drop_dict["death_rate"]

        # Get main and scaled (during test time) residual channels
        main_channels, res_channels = layers
        res_scaled = scale_activations(death_rate)(res_channels)

        # Add scaled value only if gate is open, otherwise keep untouched
        non_drop_path = add([main_channels, res_scaled])        
        ret_layer = drop_path(gate)([main_channels, non_drop_path])        
        
    else:
        ret_layer = add(layers)
        
    
    # Return final layer
    return ret_layer
    
    
def __dpn_block(x, nb_layers, nb_filter, growth_rate, drop_table,
                    smaller_filters=False, weight_decay=DEFAULT_WEIGHT_DECAY):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            Args:
                x: keras tensor
                nb_layers: the number of layers of conv_block to append to the model
                nb_filter: current number of residual_filters
                growth_rate: growth rate
                drop_table: table with death_rate and gate values for drop_paths
                smaller_filters: flag for using smaller filters later in the network
                weight_decay: weight decay factor
                
            Returns: keras tensor with updated features after nb_layers __conv_block calls
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    for i in range(nb_layers):
        cb = __conv_block(x, nb_filter + growth_rate, smaller_filters, weight_decay)

        cb_res = crop(concat_axis, 0, nb_filter)(cb)
        cb_dense = crop(concat_axis, nb_filter)(cb)
        
        if i == 0:
            total_res = res_add([x, cb_res], drop_table[i])
            total_dense = cb_dense
        else:
            total_res = res_add([total_res, cb_res], drop_table[i])
            total_dense = concatenate([total_dense, cb_dense], axis=concat_axis)

        x = concatenate([total_res, total_dense], axis=concat_axis)
        
        
    return x
    

    
def __create_dual_path_net(nb_classes, img_input, include_top, depth=DEFAULT_DEPTH, 
                        nb_dense_block=DEFAULT_NUM_BLOCKS, init_filters=DEFAULT_INIT_FILTERS, 
                        nb_layers_per_block=DEFAULT_LAYERS_PER_BLOCK, use_droppath=USE_DROPPATH,
                        dropout_rate=DEFAULT_DROPOUT_RATE, weight_decay=DEFAULT_WEIGHT_DECAY, 
                        activation='softmax'):
    ''' Build the DPN model.
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        init_filters: initial number of filters. Default -1 indicates initial number of filters is 128.
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must be nb_dense_block.
        dropout_rate: dropout rate
        weight_decay: weight decay
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
                
    Returns: keras tensor after depth number of convolutional operations in the DPN structure
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    # Get layers in each block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == nb_dense_block, 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be nb_dense_block'
        
        assert (np.sum(np.array(nb_layers)) + (nb_dense_block + 1) == depth), \
                        ('Total number of layers must add up to %d.' % depth)
    
    else:
        if nb_layers_per_block != -1:
            assert ((nb_dense_block * (nb_layers_per_block + 1)) + 1 == depth), \
						'Depth (minus nb_dense_block + 1) must be divisible by number of nb_dense_block'
                        
        else:
            assert ((depth - (nb_dense_block + 1)) % nb_dense_block == 0), \
						'Depth (minus nb_dense_block + 1) must be divisible by number of nb_dense_block'

            nb_layers_per_block = int((depth - (nb_dense_block + 1)) / nb_dense_block)

        nb_layers = [nb_layers_per_block] * nb_dense_block

        
    # Create drop table
    total_residuals = np.sum(np.array(nb_layers))
    
    if use_droppath:
        drop_table = []
        
        for _ in range(total_residuals):
            death_rate = K.variable(DEFAULT_DEATH_RATE)
            gate = K.variable(1, dtype='uint16')    
            drop_table.append({"death_rate": death_rate, "gate": gate})
            
    else:
        drop_table = [None] * total_residuals    

        
    # Initial convolution
    x = Conv2D(init_filters, (3, 3), kernel_initializer='he_uniform', 
                padding='same', name='initial_conv2D',
                use_bias=True, kernel_regularizer=l2(weight_decay))(img_input)

	
    # Check init filters to make sure they are an acceptable value
    if init_filters < 64:
        init_filters = 64
    elif init_filters % 16 != 0:
        init_filters += (16 - (init_filters % 16))
        
    
    # Add dense blocks
    prev_activations = None    
    
    dt_start = 0
    dt_end = nb_layers[0]        
    
    for block_idx in range(nb_dense_block):
        # Get next block residuals
        if init_filters <= 256:        
            next_filters = int(2 * init_filters)
        elif init_filters <= 512:
            next_filters = int(1.75 * init_filters)
        elif init_filters <= 1024:
            next_filters = int(1.5 * init_filters)
        elif init_filters <= 1536:
            next_filters = int(1.25 * init_filters)
    
        # Figure out appropriate growth rate
        filter_diff = next_filters - init_filters
        if block_idx == 0 or block_idx == (nb_dense_block - 1):
            growth_rate = 1.5
        else:
            growth_rate = 2
            
        growth_channels = int((growth_rate * filter_diff) // nb_layers[block_idx])
        
        channel_min_limit = (filter_diff * 0.05) 
        if channel_min_limit > 16:
            channel_min_limit = 16
            
        channel_max_limit = (filter_diff * 0.125)
        if channel_max_limit > 50:
            channel_max_limit = 50
        
        if growth_channels > channel_max_limit:
            growth_channels = int(channel_max_limit)
        elif growth_channels < channel_min_limit:
            growth_channels = int(channel_min_limit)
            
        # Ensure that embedding spatial area isn't too small for a 5x5 conv
        out_shape = K.int_shape(x)
        channel_dimen = concat_axis
        if concat_axis < 0:
            channel_dimen += len(out_shape)            
        
        out_shape = [ dimen for i, dimen in enumerate(out_shape) 
                        if i != 0 and i != channel_dimen ]
                        
        if out_shape[0] < 10 or out_shape[1] < 10:
            smaller_filters = True
        else:
            smaller_filters = False

        # Get dense block            
        x = __dpn_block(x, nb_layers[block_idx], init_filters, growth_channels, 
                                        drop_table[dt_start:dt_end], smaller_filters, 
                                        weight_decay=weight_decay)
		
        dt_start = int(np.sum(np.array(nb_layers[:block_idx+1])))
        dt_end = int(np.sum(np.array(nb_layers[:block_idx+2])))
        
        # Add transition_block (for every block except the last one)
        if block_idx != (nb_dense_block-1):
            x = __transition_block(prev_activations, x, next_filters, weight_decay=weight_decay)
            
            prev_activations = x            
            init_filters = next_filters

    # Average pool, apply dropout (if desired), and apply final FC layer
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)    
    x = GlobalAveragePooling2D(name="final_embeddings")(x)
    
    if include_top:
        if dropout_rate:
            x = Dropout(dropout_rate, name="final_dropout")(x)
            
        x = Dense(nb_classes, name="predictions", activation=activation, 
                        kernel_regularizer=l2(weight_decay), 
                        bias_regularizer=l2(weight_decay))(x)

    
    # Return classification logits and drop table                 
    return x, drop_table
    
