'''
NAS CNN Network (by Google) model for Keras.

Though exact implementation details were a bit ambiguous, it is currently 
the state-of-art in terms of both efficiency and accuracy in image 
classification (on a variety of datasets).  It was created using machine 
learning to construct convolutional "cells".  

# Reference:
	https://arxiv.org/pdf/1707.07012.pdf
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
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.engine.topology import get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape


USE_DROPPATH=True
DEFAULT_DEATH_RATE=0.1

ELEMENTS_PER_COMBINATION=2
COMBINATIONS_PER_LAYER=5

DEFAULT_NUM_REDUCTION=2
DEFAULT_NUM_REPEAT_VALUE=3
DEFAULT_INIT_FILTERS=128
DEFAULT_DROPOUT_RATE=0.1
DEFAULT_WEIGHT_DECAY=1E-4
DEFAULT_NUM_CLASSES=100



def NASNet(input_shape=None, num_reduction_cells=DEFAULT_NUM_REDUCTION, 
				repeat_val=DEFAULT_NUM_REPEAT_VALUE, init_filters=DEFAULT_INIT_FILTERS, 
                use_droppath=USE_DROPPATH, dropout_rate=DEFAULT_DROPOUT_RATE, 
                weight_decay=DEFAULT_WEIGHT_DECAY, include_top=True, input_tensor=None, 
                classes=DEFAULT_NUM_CLASSES, activation='softmax'):
                
    '''Instantiate a slight modification of the Dual Path Network architecture.
        The main tenants are the same, but some liberities were taken to combine 
        it with other prominent work in computer vision.  Accordingly, multiple 
        filter sizes are used for the "grouped" or separable convolutions.  Also, 
        an optional dropout is added to the final layer.  Aside from that, it is 
        very similar to the orginal construct.

        # Note
            If NAS Net is created with drop paths, the user must remember to use
            the "DynamicDropPathGates" Callback so that different paths are randomly
            closed (e.g. set to 0) for each mini-batch of training.  Also, in the 
            event that the model weights are saved, the gate must be reset to correct 
            death_rate value before using the model again (using the "set_up_death_rates" 
            function).
        
        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            num_reduction_cells: number of 2x2 strided reductions cells
            repeat_val: number of normal NAS convolutional cells surrounding the reduction cells
            init_filters: initial number of filters
                Should be 128 unless the model is extremely small or large
            use_droppath: whether to use drop paths throughout the NASNet (requires use of DropPath callback too)
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

    x, drop_table = __create_nas_cell_net(classes, img_input, include_top, init_filters,
                                num_reduction_cells, repeat_val, use_droppath, 
                                dropout_rate, weight_decay, activation)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create and return model.
    model = Model(inputs, x, name='nas_net')

    
    return model, drop_table

 
# Layer for dropping path based on a "gate" variable   
def drop_path(gate):
    def func(tensors, gate=gate):
        return K.switch(gate, tensors[1], tensors[0])
			
    return Lambda(func)  


# Layer for scaling down activations at test time
def scale_activations(drop_rate):
    def func(x, drop_rate=drop_rate):
        scale = K.ones_like(x) - drop_rate
        return K.in_test_phase(scale * x, x)
    
    return Lambda(func)
       

# Layer for eliminating path entirely       
def zero_out_path():
    def func(x):
        clear_path_mult = K.zeros_like(x)
        return clear_path_mult * x

    return Lambda(func)
    

def add_with_potential_drop_path_helper(layers, drop_table):    
    ''' Perform add operation with possibility of dropping out (e.g. excluding) every layer
    Args:
        layers: list of keras layer tensor to add together
        drop_table: table with gate and death_rate values
        
    Returns: keras tensor output after operation
    '''

    # Add gate variables and their drop rates to a global table
    final_inputs = []

    # Go through each layer and dropout based on gate value
    for layer, drop_vars in zip(layers, drop_table):
        # Get death_rate and drop gate variables from table
        gate = drop_vars["gate"]
        death_rate = drop_vars["death_rate"]
        
        # Make copy of original layer with all zeros (for drop path)
        drop_tensor = zero_out_path()(layer)
        
        # Scale inputs at test time so that the model gets the same expected sum
        scaled_layer = scale_activations(death_rate)(layer)
        
        # Add tensor to final inputs
        final_inputs.append(drop_path(gate)([drop_tensor, scaled_layer]))
    
    
    # Return all the selected layers added together
    return add(final_inputs)

    
def add_with_potential_drop_path(layers, drop_table):
    ''' Addition wrapper.  Can be either normal add or one with drop path.
    Args:
        layers: list of keras layer tensor to add together
        drop_table: table with gate and death_rate values
        
    Returns: keras tensor output after operation
    '''
    
    # Condition on whether drop_table actually in use
    if drop_table[0] is None:
        ret_layer = add(layers)
    else:
        ret_layer = add_with_potential_drop_path_helper(layers, drop_table)
    

    # Return final layer
    return ret_layer
    
    
def layer_into_spatial_and_channels(layer):
    ''' Simple helper to get certain dimensions of a layer.
    Args:
        layer: input keras tensor
        
    Returns: tuple of spatial dimension and number of channels in layer
    '''
    
    # Get height/width and channel axis
    height_or_width_dimension = 2
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    
    # Get shapes of previous layer
    layer_shape = K.int_shape(layer)
    
    layer_spatial_dimen = layer_shape[height_or_width_dimension]
    layer_channels = layer_shape[concat_axis]
    
    
    # Return dimensions
    return layer_spatial_dimen, layer_channels
    
    
def make_prev_match_cur_layer(ip_prev, desired_spatial_dimen, desired_channels, 
                weight_decay=DEFAULT_WEIGHT_DECAY):
    ''' Simple helper to ensure that both hidden layer inputs have the same dimensions.
    Args:
        ip_prev: tuple of input keras tensor from previous block and flag for input image
        desired_spatial_dimen: output spatial dimensions
        desired_channels: output channels dimensions
        weight_decay: weight decay factor
        
    Returns: keras tensor output (with either the original or new/adjusted "prev" layer)
    '''
    
    # Get channel axis
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    
    # Get shapes of previous layer
    prev_layer, prev_is_input_image = ip_prev
    prev_spatial_dimen, prev_channels = layer_into_spatial_and_channels(prev_layer)
    
    
    # Determine necessary adjustments
    prev_need_spatial_adjust = False
    prev_need_channel_adjust = False
    
    if prev_spatial_dimen != desired_spatial_dimen:
        prev_need_spatial_adjust = True

    if prev_channels != desired_channels:
        prev_need_channel_adjust = True
    
    
    # Make adjustments
    cur_input_image_val = prev_is_input_image
    if prev_need_spatial_adjust or prev_need_channel_adjust:
        # Set input image flag to false
        cur_input_image_val = False
        
        # Put old layer through necessary 1x1 convolution
        if not prev_is_input_image:
            prev_layer = Activation('relu')(prev_layer)
        
        if prev_need_spatial_adjust:
            #prev_layer = Conv2D(desired_channels, (1, 1), strides=(2,2), kernel_initializer='he_uniform', 
            prev_layer = SeparableConv2D(desired_channels, (3, 3), strides=(2,2), kernel_initializer='he_uniform', 
                                    padding='same', use_bias=True, 
                                    kernel_regularizer=l2(weight_decay))(prev_layer)
        else:
            #prev_layer = Conv2D(desired_channels, (1, 1), kernel_initializer='he_uniform', 
            prev_layer = SeparableConv2D(desired_channels, (3, 3), kernel_initializer='he_uniform', 
                                    padding='same', use_bias=True, 
                                    kernel_regularizer=l2(weight_decay))(prev_layer)        

        prev_layer = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                                            beta_regularizer=l2(weight_decay))(prev_layer)
                                            
                                            
    # Return adjusted layer
    return (prev_layer, cur_input_image_val)

    
def __normal_nas_cell(ip_prev, ip_cur, nb_filter, drop_table, weight_decay=DEFAULT_WEIGHT_DECAY):
    ''' Apply a series of operations to output of last two blocks as main conv "cell".
    Args:
        ip_prev: tuple of input keras tensor from previous block and flag for input image
        ip_cur: tuple of input keras tensor from current block and flag for input image
        nb_filter: final number of filters to output
        drop_table: table with gates values for drop_path
        weight_decay: weight decay factor
        
    Returns: keras tensor output (with same number of channels as input)
    '''

    # Get axis for channels
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    
    # Initialize general variables
    start_i = 0
    end_i = ELEMENTS_PER_COMBINATION
    total_combos = 0
    
    
    # Get RELU altered activations
    prev_layer, prev_is_input_image = ip_prev
    cur_layer, cur_is_input_image = ip_cur
    
    if prev_is_input_image:
        ip_prev_plus_relu = prev_layer

        prev_spatial, _ = layer_into_spatial_and_channels(prev_layer)
        prev_layer_alt, _ = make_prev_match_cur_layer(ip_prev, prev_spatial, nb_filter)       
        ip_prev_plus_relu_alt = Activation('relu')(prev_layer_alt)
        
    else:
        ip_prev_plus_relu = Activation('relu')(prev_layer)
        
        prev_layer_alt = prev_layer
        ip_prev_plus_relu_alt = ip_prev_plus_relu
        
        
    if cur_is_input_image:
        ip_cur_plus_relu = cur_layer
        
        cur_spatial, _ = layer_into_spatial_and_channels(cur_layer)        
        cur_layer_alt, _ = make_prev_match_cur_layer(ip_cur, cur_spatial, nb_filter)        
        ip_cur_plus_relu_alt = Activation('relu')(cur_layer_alt)

    else:
        ip_cur_plus_relu = Activation('relu')(cur_layer)
        
        cur_layer_alt = cur_layer
        ip_cur_plus_relu_alt = ip_cur_plus_relu
    
    
    # Hidden sub-block 1
    hid1_0 = SeparableConv2D(nb_filter, (3, 3), padding='same', use_bias=True)(ip_cur_plus_relu)
    hid1_0 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid1_0)
                           
    input_layers = [cur_layer_alt, hid1_0]
    assert (len(input_layers) == ELEMENTS_PER_COMBINATION)
    
    hid1 = add_with_potential_drop_path(input_layers, drop_table[start_i:end_i]) 
    
    start_i += ELEMENTS_PER_COMBINATION
    end_i += ELEMENTS_PER_COMBINATION
    total_combos += 1
    
    
    # Hidden sub-block 2
    hid2_0 = SeparableConv2D(nb_filter, (3, 3), padding='same', use_bias=True)(ip_prev_plus_relu)
    hid2_0 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid2_0)
    
    hid2_1 = SeparableConv2D(nb_filter, (5, 5), padding='same', use_bias=True)(ip_cur_plus_relu)
    hid2_1 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid2_1)

    input_layers = [hid2_0, hid2_1]
    assert (len(input_layers) == ELEMENTS_PER_COMBINATION)
                           
    hid2 = add_with_potential_drop_path(input_layers, drop_table[start_i:end_i]) 
    
    start_i += ELEMENTS_PER_COMBINATION
    end_i += ELEMENTS_PER_COMBINATION
    total_combos += 1
    
    
    # Hidden sub-block 3
    hid3_0 = AveragePooling2D((3, 3), strides=(1,1), padding='same')(ip_cur_plus_relu_alt)
    hid3_0 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid3_0)
                           
    input_layers = [prev_layer_alt, hid3_0]
    assert (len(input_layers) == ELEMENTS_PER_COMBINATION)
                           
    hid3 = add_with_potential_drop_path(input_layers, drop_table[start_i:end_i]) 
    
    start_i += ELEMENTS_PER_COMBINATION
    end_i += ELEMENTS_PER_COMBINATION
    total_combos += 1
    
    
    # Hidden sub-block 4
    hid4_average = AveragePooling2D((3, 3), strides=(1,1), padding='same')(ip_prev_plus_relu_alt)
    
    hid4_0 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid4_average)    
    hid4_1 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid4_average)    

    input_layers = [hid4_0, hid4_1]
    assert (len(input_layers) == ELEMENTS_PER_COMBINATION)
                           
    hid4 = add_with_potential_drop_path(input_layers, drop_table[start_i:end_i]) 
    
    start_i += ELEMENTS_PER_COMBINATION
    end_i += ELEMENTS_PER_COMBINATION                           
    total_combos += 1
    
    
    # Hidden sub-block 5
    hid5_0 = SeparableConv2D(nb_filter, (3, 3), padding='same', use_bias=True)(ip_prev_plus_relu)
    hid5_0 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid5_0)
    
    hid5_1 = SeparableConv2D(nb_filter, (5, 5), padding='same', use_bias=True)(ip_prev_plus_relu)
    hid5_1 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid5_1)
    
    input_layers = [hid5_0, hid5_1]
    assert (len(input_layers) == ELEMENTS_PER_COMBINATION)
    
    hid5 = add_with_potential_drop_path(input_layers, drop_table[start_i:end_i]) 
    
    start_i += ELEMENTS_PER_COMBINATION
    end_i += ELEMENTS_PER_COMBINATION                           
    total_combos += 1
    
    
    # Concatenate sub-blocks and scale them down
    assert (total_combos == COMBINATIONS_PER_LAYER)
    output = concatenate([hid1, hid2, hid3, hid4, hid5], axis=concat_axis)
    
    output = Activation('relu')(output)
    output = Conv2D(nb_filter, (1, 1), kernel_initializer='he_uniform', 
                padding='same', use_bias=True, kernel_regularizer=l2(weight_decay))(output)
    output = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(output)
                               

    # Return cell output
    return (output, False)
    
    
def __reduction_nas_cell(ip_prev, ip_cur, nb_filter, drop_table, weight_decay=DEFAULT_WEIGHT_DECAY):
    ''' Apply operations to the output of last two blocks to reduce spatial and increase channels.
    Args:
        ip_prev: tuple of input keras tensor from previous block and flag for input image
        ip_cur: tuple of input keras tensor from current block and flag for input image
        nb_filter: final number of filters to output
        drop_table: table with gates values for drop_path
        weight_decay: weight decay factor
        
    Returns: keras tensor output (with half the spatial dimensions but double the channels)
    '''

    # Get axis for channels
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1


    # Initialize general variables
    start_i = 0
    end_i = ELEMENTS_PER_COMBINATION
    total_combos = 0

    
    # Get RELU altered activations
    prev_layer, prev_is_input_image = ip_prev
    cur_layer, cur_is_input_image = ip_cur
    
    if prev_is_input_image:
        ip_prev_plus_relu = prev_layer
    else:
        ip_prev_plus_relu = Activation('relu')(prev_layer)
                
        
    if cur_is_input_image:
        ip_cur_plus_relu = cur_layer        
    else:
        ip_cur_plus_relu = Activation('relu')(cur_layer)
        
    cur_spatial, _ = layer_into_spatial_and_channels(cur_layer)
    cur_layer_alt, _ = make_prev_match_cur_layer(ip_cur, cur_spatial, nb_filter)        
    ip_cur_plus_relu_alt = Activation('relu')(cur_layer_alt)

    
    # Hidden sub-block 1    
    hid1_0 = SeparableConv2D(nb_filter, (5, 5), strides=(2, 2), padding='same', 
                                use_bias=True)(ip_cur_plus_relu)
    hid1_0 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid1_0)

    hid1_sep_7 = SeparableConv2D(nb_filter, (7, 7), strides=(2, 2), padding='same', 
                                use_bias=True)(ip_prev_plus_relu)
    hid1_1 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid1_sep_7)
                           
    input_layers = [hid1_0, hid1_1]
    assert (len(input_layers) == ELEMENTS_PER_COMBINATION)
                           
    hid1 = add_with_potential_drop_path(input_layers, drop_table[start_i:end_i]) 
    
    start_i += ELEMENTS_PER_COMBINATION
    end_i += ELEMENTS_PER_COMBINATION                           
    total_combos += 1          
    
    
    # Hidden sub-block 2
    hid2_max_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(ip_cur_plus_relu_alt)
    
    hid2_0 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid2_max_3)
    hid2_1 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid1_sep_7)
                           
    input_layers = [hid2_0, hid2_1]
    assert (len(input_layers) == ELEMENTS_PER_COMBINATION)
                           
    hid2 = add_with_potential_drop_path(input_layers, drop_table[start_i:end_i]) 
    
    start_i += ELEMENTS_PER_COMBINATION
    end_i += ELEMENTS_PER_COMBINATION                           
    total_combos += 1
    
    
    # Hidden sub-block 3
    hid3_0 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(ip_cur_plus_relu_alt)
    hid3_0 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid3_0)    

    hid3_1 = SeparableConv2D(nb_filter, (5, 5), strides=(2, 2), padding='same', 
                                use_bias=True)(ip_prev_plus_relu)
    hid3_1 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid3_1)
           
    input_layers = [hid3_0, hid3_1]
    assert (len(input_layers) == ELEMENTS_PER_COMBINATION)
                           
    hid3 = add_with_potential_drop_path(input_layers, drop_table[start_i:end_i]) 
    
    start_i += ELEMENTS_PER_COMBINATION
    end_i += ELEMENTS_PER_COMBINATION                           
    total_combos += 1
    
    
    # Hidden sub-block 4
    hid4_0 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid2_max_3)
    
    hid4_relu = Activation('relu')(hid1)
    hid4_1 = SeparableConv2D(nb_filter, (3, 3), padding='same', use_bias=True)(hid4_relu)
    hid4_1 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid4_1)                          
           
    input_layers = [hid4_0, hid4_1]
    assert (len(input_layers) == ELEMENTS_PER_COMBINATION)
                           
    hid4 = add_with_potential_drop_path(input_layers, drop_table[start_i:end_i]) 
    
    start_i += ELEMENTS_PER_COMBINATION
    end_i += ELEMENTS_PER_COMBINATION                           
    total_combos += 1
    
    
    # Hidden sub-block 5
    hid5_0 = AveragePooling2D((3, 3), strides=(1,1), padding='same')(hid4_relu)
    hid5_0 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(hid5_0)

    input_layers = [hid2, hid5_0]
    assert (len(input_layers) == ELEMENTS_PER_COMBINATION)
                           
    hid5 = add_with_potential_drop_path(input_layers, drop_table[start_i:end_i]) 
    
    start_i += ELEMENTS_PER_COMBINATION
    end_i += ELEMENTS_PER_COMBINATION                           
    total_combos += 1
    
    
    # Concatenate sub-blocks and scale them down
    assert (total_combos == COMBINATIONS_PER_LAYER)
    output = concatenate([hid3, hid4, hid5], axis=concat_axis)
    
    output = Activation('relu')(output)
    output = Conv2D(nb_filter, (1, 1), kernel_initializer='he_uniform', 
                padding='same', use_bias=True, kernel_regularizer=l2(weight_decay))(output)
    output = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(output)

                               
    # Return cell output
    return (output, False)
            
                                
def __create_nas_cell_net(nb_classes, img_input, include_top, init_filters=DEFAULT_INIT_FILTERS, 
                        num_reduction_cells=DEFAULT_NUM_REDUCTION, repeat_val=DEFAULT_NUM_REPEAT_VALUE,
                        use_droppath=USE_DROPPATH, dropout_rate=DEFAULT_DROPOUT_RATE, 
                        weight_decay=DEFAULT_WEIGHT_DECAY, activation='softmax'):
    ''' Build the NAS Cell model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        init_filters: initial number of filters. Default -1 indicates initial number of filters is 128.
        num_reduction_cells: number of 2x2 strided reductions cells
        repeat_val: number of normal NAS convolutional cells surrounding the reduction cells        
        dropout_rate: dropout rate
        weight_decay: weight decay
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
                
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    
    # Get channel axis
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    
    # Create drop table
    total_elements_per_NAS_cell = COMBINATIONS_PER_LAYER * ELEMENTS_PER_COMBINATION
    total_cells = num_reduction_cells + ((num_reduction_cells + 1) * repeat_val)
    total_elements = total_cells * total_elements_per_NAS_cell
    
    if use_droppath:
        drop_table = []
        
        for _ in range(total_elements):
            death_rate = K.variable(DEFAULT_DEATH_RATE)
            gate = K.variable(1, dtype='uint16')    
            drop_table.append({"death_rate": death_rate, "gate": gate})
            
    else:
        drop_table = [None] * total_elements    

        
    # Begin the process of Normal NAS Cell and Reduction NAS cell sequences        
    dt_start = 0
    dt_end = total_elements_per_NAS_cell    

    cur_hid = (img_input, True)
    prev_hid = (img_input, True)
    
    num_channels = init_filters
    for i in range(num_reduction_cells + 1):
        # Iterate N times through normal cell
        for _ in range(repeat_val):
            cur_spatial, cur_channels = layer_into_spatial_and_channels(cur_hid[0])
            prev_hid = make_prev_match_cur_layer(prev_hid, cur_spatial, cur_channels)

            tmp_prev_hid = cur_hid            
            cur_hid = __normal_nas_cell(prev_hid, cur_hid, num_channels, 
                                                drop_table[dt_start:dt_end], weight_decay)
            prev_hid = tmp_prev_hid
            
            dt_start += total_elements_per_NAS_cell
            dt_end += total_elements_per_NAS_cell

        # Double number of channels and pass through reduction cell
        num_channels = int(num_channels * 2.5)
        if i < num_reduction_cells:
            cur_spatial, cur_channels = layer_into_spatial_and_channels(cur_hid[0])
            prev_hid = make_prev_match_cur_layer(prev_hid, cur_spatial, cur_channels)

            tmp_prev_hid = cur_hid            
            cur_hid = __reduction_nas_cell(prev_hid, cur_hid, num_channels, 
                                                drop_table[dt_start:dt_end], weight_decay)
            prev_hid = tmp_prev_hid
            
            dt_start += total_elements_per_NAS_cell
            dt_end += total_elements_per_NAS_cell
            
            
    # Average pool, apply dropout (if desired), and apply final FC layer
    x = cur_hid[0]
    
    x = Activation('relu')(x)    
    x = GlobalAveragePooling2D(name="final_embeddings")(x)
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
    
    if include_top:
        if dropout_rate:
            x = Dropout(dropout_rate, name="final_dropout")(x)
            
        x = Dense(nb_classes, activation=activation, kernel_regularizer=l2(weight_decay), 
                        bias_regularizer=l2(weight_decay))(x)

                        
    # Return classification logits and drop table                 
    return x, drop_table
