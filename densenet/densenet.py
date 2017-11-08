'''
Densely Connected Network model for Keras.

This model is unique in that it does not use residual connections but rather,
it saves a same set of channels for each layer and leverages them as inputs to 
subsequent layers.  This has the effect of shortening the distance between layers 
and therefore improving gradient flow.  It is one of the most influential models 
in the past several years in computer vision. 

# Note:
    While the code for a DenseNet model will always look similar regardless of 
    particular implementation choices, it is still prudent to acknowledge the 
    heavy influence of the GitHub repo (titu1994/DenseNet) on this work.  While 
    there are substanial changes from the original code and differences in low-
	level details, the structure of the functionality remains similar in spirit.

# Reference:
    https://arxiv.org/pdf/1608.06993.pdf
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import math

import numpy as np
import tensorflow as tf

import keras
import keras.backend as K

from keras import metrics
from keras.optimizers import *
from keras.models import Model

from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape


DEFAULT_FINAL_DROPOUT_RATE=0.1

TF_WEIGHTS_PATH = 'https://github.com/titu1994/DenseNet/releases/download/v2.0/DenseNet-40-12-Tensorflow-Backend-TF-dim-ordering.h5'



def DenseNet(input_shape=None, depth=82, nb_dense_block=3, growth_rate=16, nb_filter=32, 
				nb_layers_per_block=-1, bottleneck=True, reduction=0.25, dropout_rate=0.0, 
				final_dropout=DEFAULT_FINAL_DROPOUT_RATE, weight_decay=1E-4, include_top=True, 
				weights=None, input_tensor=None, classes=100, activation='softmax'):
			 
    '''Instantiate the DenseNet architecture, optionally loading weights pre-trained
        on CIFAR-100. Note that when using TensorFlow, for best performance you should 
        set `image_data_format='channels_last'` in your Keras config at ~/.keras/keras.json.
        
        The model and the weights are compatible with both TensorFlow and Theano. 
        
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
            depth: number or layers in the DenseNet
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. -1 indicates initial
                number of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the network depth.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            bottleneck: flag to add bottleneck blocks in between dense blocks
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            final_dropout: dropout rate of final FC layer
            weight_decay: weight decay factor
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                'cifar100' (pre-training on CIFAR-100).
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

    # Check parameters	
    if weights not in {'cifar100', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-100).')

    if weights == 'cifar100' and include_top and classes != 100:
        raise ValueError('If using `weights` as CIFAR 100 with `include_top`'
                         ' as true, `classes` should be 100')

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

            
    # Create DN output
    x = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck, 
                           reduction, dropout_rate, final_dropout, weight_decay, 
                           activation)

                           
    # Ensure that the model takes into account any potential predecessors of `input_tensor`
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
        
    # Turn DN output into model
    model = Model(inputs, x, name='densenet')


    # Load weights, if necessary
    if weights == 'cifar100':
        if ((depth == 40) and (nb_dense_block == 3) and (growth_rate == 12) and 
                (nb_filter == 16) and (bottleneck is False) and (reduction == 0.0) 
                and (dropout_rate == 0.0) and (final_dropout == DEFAULT_FINAL_DROPOUT_RATE) 
                and (weight_decay == 1E-4)):
            
            # Since default parameters match, it is possible to load weights
            assert K.image_data_format() != 'channels_first', \
                        'To load pre-trained weights, Keras config file must set to use "channels_last".'
            
            if include_top:
                weights_path = get_file('densenet_40_12_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models_weights')
            else:
                weights_path = get_file('densenet_40_12_tf_dim_ordering_tf_kernels_no_top.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models_weights')

            model.load_weights(weights_path)

            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
                            

    # Return either pre-trained or randomly initialized model
    return model


def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if bottleneck:
        x = Conv2D(nb_filter * 4, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=True,
                   kernel_regularizer=l2(weight_decay))(ip)

        x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
    else:
        x = ip
        
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=True,
               kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __transition_block(ip, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        
    Returns: keras tensor, after applying BN, Relu-Conv, Dropout, Maxpool ops
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_uniform', padding='same', 
                use_bias=True, kernel_regularizer=l2(weight_decay))(ip)
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
		
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, 
                weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=concat_axis)
        
        nb_filter += growth_rate

    return x, nb_filter


def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, 
						growth_rate=12, nb_filter=-1, nb_layers_per_block=-1, bottleneck=False, 
						reduction=0.0, dropout_rate=None, final_dropout=DEFAULT_FINAL_DROPOUT_RATE, 
                        weight_decay=1E-4, activation='softmax'):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
					be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: general dropout rate
        final_dropout: dropout rate of final FC layer
        weight_decay: weight decay
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
                
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'Reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == nb_dense_block, 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be nb_dense_block'
        
        assert (np.sum(np.array(nb_layers)) + (nb_dense_block + 1) == depth), \
                        ('Total number of layers must add up to %d.' % depth)
        
    else:
        if nb_layers_per_block == -1:
            assert ((depth - (nb_dense_block + 1)) % nb_dense_block == 0), \
						'Depth (minus nb_dense_block + 1) must be divisible by number of nb_dense_block'

            nb_layers_per_block = int((depth - (nb_dense_block + 1)) / nb_dense_block)

        nb_layers = [nb_layers_per_block] * nb_dense_block

    if bottleneck:
        for layer in nb_layers:
            assert layer % 2 == 0, 'Blocks with bottleneck must have even number of layers within the block'
            
        nb_layers = [ int(layer / 2) for layer in nb_layers ]

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = int(2 * growth_rate)

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_uniform', padding='same', name='initial_conv2D',
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
						   beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
		
    # Add dense blocks
    for block_idx in range(nb_dense_block):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, 
                                        bottleneck=bottleneck, dropout_rate=dropout_rate, 
                                        weight_decay=weight_decay)
		
        # Add transition_block (for every block except the last one)
        if block_idx != (nb_dense_block-1):
            x = __transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression)

    # Once done, take an average pool of the spatial dimensions        
    x = GlobalAveragePooling2D(name="final_embeddings")(x)
    
    # Add final FC layer if requested
    if include_top:
        if final_dropout:
            x = Dropout(final_dropout, name="final_dropout")(x)
            	
        x = Dense(nb_classes, activation=activation, kernel_regularizer=l2(weight_decay), 
                        bias_regularizer=l2(weight_decay))(x)

                        
    # Return final embeddings                     
    return x