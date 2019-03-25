'''
Densely Connected Network model for Keras.

This model is unique in that it does not use residual connections but rather,
it saves a same set of channels for each layer and leverages them as inputs to 
subsequent layers.  This has the effect of shortening the distance between layers 
and therefore improving gradient flow.  It is one of the most influential models 
in the past several years in computer vision. 

# Note:
    1. While the code for a DenseNet model will always look similar regardless of 
    particular implementation choices, it is still prudent to acknowledge the 
    heavy influence of the GitHub repo (titu1994/DenseNet) on this work.  While 
    there are substanial changes from the original code and differences in low-
	level details, the structure of the functionality remains similar in spirit.
    
    2. These are the test results for each pretrained model on the CIFAR-100 
    dataset.  They are described in the following format -- 
    (validation_loss, accuracy, top_5_accuracy):
        Small model: 1.7225665634155274, 0.735, 0.917
        Medium model: 1.4375544189453124, 0.7656, 0.9296
        Large model: 1.5171770874023438, 0.761, 0.9268
    
# Reference:
    https://arxiv.org/pdf/1608.06993.pdf
'''

# Import statements
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import math, sys, os

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
from keras_applications.imagenet_utils import _obtain_input_shape

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../")     

from dl_utilities.layers import general as dl_layers    # Requires the 'sys.path' call above this



# Define global variables
DEFAULT_FINAL_DROPOUT_RATE=0.1
DN_WEIGHTS_DIR_PATH = 'https://github.com/alijkhalil/state_of_art_cnns/raw/master/densenet/pretrained_weights/'

small_blocks = [10, 16, 24, 12]
small_dense_block = len(small_blocks)
small_depth = np.sum(np.array(small_blocks)) + small_dense_block + 1
small_growth = 16
small_dropout = DEFAULT_FINAL_DROPOUT_RATE

SMALL_WEIGHT_NAME = ('DenseNet-%d-%d-TF-Backend.h5' % 
                        (small_depth, small_growth))
SMALL_KWARGS = {'input_shape': (32, 32, 3), 'depth': small_depth, 'nb_dense_block': small_dense_block, 
                'growth_rate': small_growth, 'nb_filter': 32, 'nb_layers_per_block': small_blocks,  
                'bottleneck': True, 'reduction': 0.3, 'dropout_rate': 0.0, 
                'final_dropout': small_dropout, 'weight_decay': 1E-4, 'include_top': True, 
                'weights': None, 'input_tensor': None, 'classes': 100, 'activation': 'softmax'}
                
                
med_blocks = [12, 24, 38, 14]
med_dense_block = len(med_blocks)
med_depth = np.sum(np.array(med_blocks)) + med_dense_block + 1
med_growth = 22
med_dropout = DEFAULT_FINAL_DROPOUT_RATE * 2

MED_WEIGHT_NAME = ('DenseNet-%d-%d-TF-Backend.h5' % 
                        (med_depth, med_growth))
MED_KWARGS = {'input_shape': (32, 32, 3), 'depth': med_depth, 'nb_dense_block': med_dense_block, 
                'growth_rate': med_growth, 'nb_filter': 32, 'nb_layers_per_block': med_blocks, 
                'bottleneck': True, 'reduction': 0.3, 'dropout_rate': 0.0, 
                'final_dropout': med_dropout, 'weight_decay': 1E-4, 'include_top': True, 
                'weights': None, 'input_tensor': None, 'classes': 100, 'activation': 'softmax'}

                
large_blocks = [12, 24, 48, 32, 16]
large_dense_block = len(large_blocks)
large_depth = np.sum(np.array(large_blocks)) + large_dense_block + 1
large_growth = 30
large_dropout = DEFAULT_FINAL_DROPOUT_RATE * 4

LARGE_WEIGHT_NAME = ('DenseNet-%d-%d-TF-Backend.h5' % 
                        (large_depth, large_growth))
LARGE_KWARGS = {'input_shape': (32, 32, 3), 'depth': large_depth, 'nb_dense_block': large_dense_block, 
                'growth_rate': large_growth, 'nb_filter': 32, 'nb_layers_per_block': large_blocks, 
                'bottleneck': True, 'reduction': 0.6, 'dropout_rate': 0.0, 
                'final_dropout': large_dropout, 'weight_decay': 1E-4, 'include_top': True, 
                'weights': None, 'input_tensor': None, 'classes': 100, 'activation': 'softmax'}

                
                
# Main functionality
def DenseNet(input_shape=None, depth=82, nb_dense_block=3, growth_rate=16, nb_filter=32, 
				nb_layers_per_block=-1, bottleneck=True, reduction=0.25, dropout_rate=0.0, 
				final_dropout=DEFAULT_FINAL_DROPOUT_RATE, weight_decay=1E-4, include_top=True, 
				weights=None, input_tensor=None, classes=100, activation='softmax', **kwargs):
			 
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
            weights: either `None` (for random initialization) or one of the following 
                pre-trained models (will disgard all other DenseNet constructor parameters):
                    'small-cifar-100' (pre-training on CIFAR-100 with model size of .8M parameters).
                    'medium-cifar-100' (pre-training on CIFAR-100 with model size of 3M parameters).
                    'large-cifar-100' (pre-training on CIFAR-100 with model size of 5M parameters).
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
    weight_options = ['small-cifar-100', 'medium-cifar-100', 'large-cifar-100', None]
    if weights not in weight_options:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `<size>-cifar-100` '
                         '(for a model pre-trained on CIFAR-100).')

    if weights in weight_options[:-1]:
        input_tensor = None
        input_shape = (32, 32, 3)
        
        if weights == weight_options[0]:
            kwarg_val = SMALL_KWARGS
            weight_path = DN_WEIGHTS_DIR_PATH + SMALL_WEIGHT_NAME
            tmp_path = 'DN_small.h5'
            
        elif weights == weight_options[1]:
            kwarg_val = MED_KWARGS
            weight_path = DN_WEIGHTS_DIR_PATH + MED_WEIGHT_NAME        
            tmp_path = 'DN_med.h5'
            
        else:
            kwarg_val = LARGE_KWARGS
            weight_path = DN_WEIGHTS_DIR_PATH + LARGE_WEIGHT_NAME        
            tmp_path = 'DN_large.h5'
        
        if include_top and classes != 100:
            raise ValueError('If using `weights` as CIFAR 100 with `include_top`'
                             ' as true, `classes` should be 100')

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

        
    # Determine proper input shape
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

            
    # Create DN output
    if weights in weight_options[:-1]:
        classes = kwarg_val['classes']
        include_top = kwarg_val['include_top']
        depth = kwarg_val['depth']
        nb_dense_block = kwarg_val['nb_dense_block']
        growth_rate = kwarg_val['growth_rate']
        nb_filter = kwarg_val['nb_filter']
        nb_layers_per_block = kwarg_val['nb_layers_per_block']
        bottleneck = kwarg_val['bottleneck']
        reduction = kwarg_val['reduction']
        dropout_rate = kwarg_val['dropout_rate']
        final_dropout = kwarg_val['final_dropout']
        weight_decay = kwarg_val['weight_decay']
        activation = kwarg_val['activation']
                            
    out = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck, 
                           reduction, dropout_rate, final_dropout, weight_decay, 
                           activation)

                           
    # Ensure that the model takes into account any potential predecessors of `input_tensor`
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
        
    # Turn DN output into model
    model = Model(inputs, out, name='densenet')


    # Load weights if necessary
    if weights in weight_options[:-1]:
        # Since default parameters match, it is possible to load weights
        assert K.image_data_format() != 'channels_first', \
                    'To load pre-trained weights, Keras config file must set to use "channels_last".'

        # Load weights                
        weights_path = get_file(tmp_path,
                                weight_path,
                                cache_subdir='tmp_weights')
                
        model.load_weights(weights_path)
        
        
    # Exclude top layer if not desired    
    final_embed = model.get_layer('final_embeddings')    
    if not include_top:
        model = Model(inputs=model.input, outputs=final_embed.output)

        
    # Convert if using Theano    
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
        x = dl_layers.custom_swish()(x)
    else:     
        x = ip
        
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=True,
                    kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                                beta_regularizer=l2(weight_decay))(x)
    x = dl_layers.custom_swish()(x)
    
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
    x = dl_layers.custom_swish()(x)
        
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
    
    # Pass thru SE block
    x = dl_layers.se_block(4)(x)
    
    # Get new layers
    init_layers = K.int_shape(x)[concat_axis]
    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=concat_axis)
        
        nb_filter += growth_rate

        
    # Pass new layers thru SE block    
    orig_l = dl_layers.crop(0, init_layers, concat_axis)(x)
    new_l = dl_layers.crop(init_layers, dimension=concat_axis)(x)
    
    new_after_se = dl_layers.se_block(8)(new_l)
    x = concatenate([orig_l, new_after_se], axis=concat_axis)
    
    # Return channels
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

    # Get layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == nb_dense_block, 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be nb_dense_block'
        
        assert (np.sum(np.array(nb_layers)) + (nb_dense_block + 1) == depth), \
                        ('Total number of layers must add up to %d. Currently is %d.' % 
                            (depth, np.sum(np.array(nb_layers)) + (nb_dense_block + 1)))
                            
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

        
    # Compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = int(2 * growth_rate)

    # Compute compression factor
    compression = 1.0 - reduction

    
    # Initial convolution
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_uniform', padding='same', name='initial_conv2D',
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
						   beta_regularizer=l2(weight_decay))(x)
    x = dl_layers.custom_swish()(x)
		
        
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
            
    
    # Add final FC layer if requested
    if include_top:
        # Once done, take an average pool of the spatial dimensions        
        x = GlobalAveragePooling2D(name="final_embeddings")(x)    
        
        if final_dropout:
            x = Dropout(final_dropout, name="final_dropout")(x)
            	
        x = Dense(nb_classes, activation=activation, kernel_regularizer=l2(weight_decay), 
                        bias_regularizer=l2(weight_decay))(x)

                        
    # Return final embeddings                     
    return x

    
    
    
    
    
    

    
    
#############  TRAINING/TESTING ROUTINE FOR PRE-TRAINED MODELS ############
'''
if __name__ == '__main__':

    # Testing specific import statements
    from keras.datasets import cifar100
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint

    from dl_utilities.callbacks import callback_utils as cb_utils
    from dl_utilities.datasets import dataset_utils as ds_utils    
    
    
    # Get training/test data and normalize/standardize it
    num_classes = 100
    
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train, x_test = ds_utils.normal_image_preprocess(x_train, x_test)
                
	# Convert class vectors to sparse/binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    # Initialize model and set model-specific variables
    weight_dirname = "./pretrained_weights/"
    model = DenseNet(**LARGE_KWARGS)          # MODIFY
    
    # Print model summary
    model.summary()

    
    # Set up callbacks depending on options
    num_snapshots = 3
    for i in range(num_snapshots):
        # Declare variables
        init_lr_val = float(0.2 - (i * 0.025))
        batch_size = 128
        num_epochs = 175
       
       
        # Set up image augmentation generator
        global_image_aug = ImageDataGenerator(
                                        rotation_range=(15 + (i * 2.5)), 
                                        width_shift_range=((6. + (i * 2.)) / x_train.shape[2]), 
                                        height_shift_range=((6. + (i * 2.)) / x_train.shape[1]), 
                                        horizontal_flip=True, 
                                        zoom_range=(0.15 + (i * 0.025)))
                
        
        # Load pre-existing weights       
        weight_path = weight_dirname + LARGE_WEIGHT_NAME    # MODIFY
        
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)

            
        # Set up callbacks (starting with decreasing LR)
        callbacks = [ cb_utils.CosineLRScheduler(init_lr_val, num_epochs) ]
            
        # Add weight saving callback            
        callbacks.append(ModelCheckpoint(weight_path, 
                            monitor="acc", period=int(num_epochs // 5),
                            save_best_only=False, save_weights_only=True))
                                
                                
        # Add increasing Dropout callback
        callbacks.append(cb_utils.DynamicDropoutWeights(large_dropout))   # MODIFY
        
        
        # Compile model and conduct training
        model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=init_lr_val), 
                            metrics=['accuracy', metrics.top_k_categorical_accuracy])   

        hist = model.fit_generator(
                        global_image_aug.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=(x_train.shape[0] // batch_size),
                        epochs=num_epochs, 
                        initial_epoch=0,
                        callbacks=callbacks,
                        validation_data=(x_test, y_test))
                   
        # Print metrics                
        print(model.metrics_names)
        print(model.evaluate(x_test, y_test, verbose=0))


    # Return successfully
    exit(0)
'''
