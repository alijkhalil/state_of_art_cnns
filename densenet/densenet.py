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
    heavy influence of the GitHub repo (titu1994/DenseNet) on this work.  There 
    have been a large number of substanial changes made to the original code, 
    but the core of the code remains the same in spirit.

# Reference:
    https://arxiv.org/pdf/1608.06993.pdf
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


DEFAULT_FINAL_DROPOUT_RATE=0.2

TH_WEIGHTS_PATH = 'https://github.com/titu1994/DenseNet/releases/download/v2.0/DenseNet-40-12-Theano-Backend-TH-dim-ordering.h5'
TF_WEIGHTS_PATH = 'https://github.com/titu1994/DenseNet/releases/download/v2.0/DenseNet-40-12-Tensorflow-Backend-TF-dim-ordering.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/titu1994/DenseNet/releases/download/v2.0/DenseNet-40-12-Theano-Backend-TH-dim-ordering-no-top.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/titu1994/DenseNet/releases/download/v2.0/DenseNet-40-12-Tensorflow-Backend-TF-dim-ordering-no-top.h5'



def DenseNet(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, 
				nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=0.0, 
				final_dropout=0.0, weight_decay=1E-4, include_top=True, weights='cifar10', 
				input_tensor=None, classes=100, activation='softmax'):
			 
    '''Instantiate the DenseNet architecture, optionally loading weights pre-trained
        on CIFAR-10. Note that when using TensorFlow, for best performance you should 
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
                'cifar10' (pre-training on CIFAR-10)..
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

    if weights not in {'cifar10', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    if weights == 'cifar10' and include_top and classes != 10:
        raise ValueError('If using `weights` as CIFAR 10 with `include_top`'
                         ' as true, `classes` should be 10')

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

    x = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck, 
						   reduction, dropout_rate, final_dropout, weight_decay, 
						   activation)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='densenet')

    # load weights
    if weights == 'cifar10':
        if (depth == 40) and (nb_dense_block == 3) and (growth_rate == 12) and (nb_filter == 16) and \
                (bottleneck is False) and (reduction == 0.0) and (dropout_rate == 0.0) and (weight_decay == 1E-4):
            
			# Default parameters match. Weights for this model exist:
            if K.image_data_format() == 'channels_first':
                if include_top:
                    weights_path = get_file('densenet_40_12_th_dim_ordering_th_kernels.h5',
                                            TH_WEIGHTS_PATH,
                                            cache_subdir='models')
                else:
                    weights_path = get_file('densenet_40_12_th_dim_ordering_th_kernels_no_top.h5',
                                            TH_WEIGHTS_PATH_NO_TOP,
                                            cache_subdir='models')

                model.load_weights(weights_path)

                if K.backend() == 'tensorflow':
                    warnings.warn('You are using the TensorFlow backend, yet you '
                                  'are using the Theano '
                                  'image dimension ordering convention '
                                  '(`image_data_format="channels_first"`). '
                                  'For best performance, set '
                                  '`image_data_format="channels_last"` in '
                                  'your Keras config '
                                  'at ~/.keras/keras.json.')
                    convert_all_kernels_in_model(model)
            else:
                if include_top:
                    weights_path = get_file('densenet_40_12_tf_dim_ordering_tf_kernels.h5',
                                            TF_WEIGHTS_PATH,
                                            cache_subdir='models')
                else:
                    weights_path = get_file('densenet_40_12_tf_dim_ordering_tf_kernels_no_top.h5',
                                            TF_WEIGHTS_PATH_NO_TOP,
                                            cache_subdir='models')

                model.load_weights(weights_path)

                if K.backend() == 'theano':
                    convert_all_kernels_in_model(model)

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
        
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
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
						reduction=0.0, dropout_rate=None, final_dropout=None, weight_decay=1E-4, 
						activation='softmax'):
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
        dropout_rate: dropout rate
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
	
	
if __name__ == '__main__':

    # Set up TF session
    num_cores = 6
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,
                                intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores)

    # https://github.com/tensorflow/tensorflow/blob/30b52579f6d66071ac7cdc7179e2c4aae3c9cb88/tensorflow/core/protobuf/config.proto#L35
    # If true, the allocator does not pre-allocate the entire specified
    # GPU memory region, instead starting small and growing as needed.
    config.gpu_options.allow_growth=True

    sess = tf.Session(config=config)
    K.set_session(sess)


    # Get training/test data and normalize/standardize it
    num_classes = 100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    for i in range(x_train.shape[-1]):
        mean_val = np.mean(x_train[:, :, :, i])

        x_train[:, :, :, i] -= mean_val
        x_train[:, :, :, i] /= 128

        x_test[:, :, :, i] -= mean_val
        x_test[:, :, :, i] /= 128
                
                
	# Convert class vectors to sparse/binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
	

	# Set up image augmentation generator
    global_image_aug = ImageDataGenerator(
                                rotation_range=10, 
                                width_shift_range=(4. / x_train.shape[2]), 
                                height_shift_range=(4. / x_train.shape[1]), 
                                horizontal_flip=True, 
                                zoom_range=0.15)
    
    
    # Set up cosine annealing LR schedule callback
    init_lr_val = 0.125
    num_epochs = 100

    def get_cosine_scaler(base_val, cur_iter, total_iter):
        if cur_iter < total_iter:
            return (0.5 * base_val * (math.cos(math.pi * 
                            (cur_iter % total_iter) / total_iter) + 1))
        else:
            return 0
            
    def variable_epochs_cos_scheduler(init_lr=init_lr_val, total_epochs=num_epochs):
        def variable_epochs_cos_scheduler_helper(cur_epoch):
            return get_cosine_scaler(init_lr, cur_epoch, total_epochs)
            
        return variable_epochs_cos_scheduler_helper    
    
    callbacks = [ LearningRateScheduler(variable_epochs_cos_scheduler()) ]
    
    
    # Set up increasing Dropout callback
    final_dropout = DEFAULT_DROPOUT_RATE
        
    class DynamicDropoutWeights(Callback):
        def __init__(self, final_dropout):
            super(DynamicDropoutWeights, self).__init__()

            if final_dropout < 0.3:
                range_val = final_dropout * 0.375
            elif final_dropout < 0.6:
                range_val = 0.175
            else:
                range_val = 0.25
             
            self.final_dropout = final_dropout 
            self.range = range_val
            
        def on_epoch_begin(self, epoch, logs={}):
            # At start of every epoch, slowly increase dropout towards final value                                                        
            total_epoch = self.params["epochs"]
            subtract_val = get_cosine_scaler(self.range, (epoch + 1), total_epoch)            

            dropout_layer = self.model.get_layer("final_dropout")            
            dropout_layer.rate = (self.final_dropout - subtract_val)

    callbacks.append(DynamicDropoutWeights(final_dropout))
            
            
    # Initialize model and conduct training
    model = DenseNet(x_train.shape[1:], depth=124, 
                        nb_dense_block=3, nb_layers_per_block=[24, 64, 32],
						bottleneck=True, reduction=0.4, growth_rate=20, weights=None, 
						dropout_rate=0.0, final_dropout=DEFAULT_FINAL_DROPOUT_RATE, 
						classes=num_classes)
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                        optimizer=SGD(lr=init_lr_val), 
                        metrics=['accuracy', metrics.top_k_categorical_accuracy])   
    
    batch_size = 64
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