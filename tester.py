# Import statements
import argparse

import tensorflow as tf
import numpy as np

from nas_net import nas_net
from densenet import densenet
from dual_path_net import dpn
from polynet import polynet

import keras.backend as K

from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from dl_utilities.callbacks import callback_utils as cb_utils
from dl_utilities.datasets import dataset_utils as ds_utils



# Model options 
DPN_KEYWORD='dpn'
NAS_KEYWORD='nasnet'
DN_KEYWORD='densenet'
POLY_KEYWORD='polynet'

DROPPATH_MODELS=[ DPN_KEYWORD, NAS_KEYWORD ]



#############   MAIN ROUTINE   #############	
if __name__ == '__main__':

    # Set up argument parser 
	#
	# Format: 
	#		tester.py <model_type> [--dropout [<float>]] [--droppath [<float>]] 
	#							[--lr [<learning_rate>]] 
	#							[--epochs [<num_of_training_epochs>]] [--save_weights]
	#
    parser = argparse.ArgumentParser(description='Test module for state-of-the-art CNNs.')
    
    parser.add_argument('model_type', choices=[DPN_KEYWORD, NAS_KEYWORD, DN_KEYWORD, POLY_KEYWORD],
                            help='CNN model of choice (required)')
    parser.add_argument('--dropout', nargs='?', default=0.0, const=0.1, type=float, metavar='final_dropout_rate',
                            help='apply dropout to final model layer at this rate (flag alone defaults to 0.1)')
    parser.add_argument('--droppath', nargs='?', default=0.0, const=0.15, type=float, metavar='droppath_rate',
                            help='if model contains "res" connections, apply drop to them at this rate (flag alone defaults to 0.2)')                            
    parser.add_argument('--lr', nargs='?', default=0.125, const=0.125, type=float, metavar='learning_rate',
                            help='learning rate for training with cosine annealing SGD (no flag defaults to 0.125)')                            
    parser.add_argument('--epochs', nargs='?', default=100, const=100, type=int, metavar='num_training_epochs',
                            help='number of training epochs on the CIFAR100 dataset (no flag defaults to 100)')                            
    parser.add_argument('--save_weights', action='store_true', help='flag to save weights in the following format: ' \
                                    '<model_dir>/weights-cifar100-<dropout>-<droppath>-<lr>-<epochs>.hdf5')                            							
    
    args = parser.parse_args()

    
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
    x_train, x_test = ds_utils.normal_image_preprocess(x_train, x_test)
                
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
    
    
    # Initialize model and set model-specific variables
    model_type = args.model_type
    
    final_dropout = args.dropout
    final_droppath = args.droppath
    
    if model_type == NAS_KEYWORD:
        weight_dirname = "./nas_net/"
        model, drop_table = nas_net.NASNet(x_train.shape[1:], init_filters=48, repeat_val=3)

        total_elements_per_NAS_cell = (nas_net.COMBINATIONS_PER_LAYER * 
                                        nas_net.ELEMENTS_PER_COMBINATION)
        total_NAS_cells = int(len(drop_table) / total_elements_per_NAS_cell)
        gates_per_layer = [total_elements_per_NAS_cell] * total_NAS_cells

    elif model_type == DN_KEYWORD:
        weight_dirname = "./densenet/"
        model = densenet.DenseNet(x_train.shape[1:], depth=88, 
                            nb_dense_block=3, nb_layers_per_block=[16, 48, 20],
                            bottleneck=True, reduction=0.4, growth_rate=24, weights=None, 
                            dropout_rate=0.0, final_dropout=final_dropout, 
                            classes=num_classes)    

        
    elif model_type == POLY_KEYWORD:
        weight_dirname = "./polynet/"
        model = polynet.PolyNet(x_train.shape[1:], init_nb_filters=88, 
                            final_dropout=final_dropout, 
                            classes=num_classes)    

    elif model_type == DPN_KEYWORD:
        weight_dirname = "./dual_path_net/"
        model, drop_table = dpn.DualPathNetwork(x_train.shape[1:], init_filters=64)

        total_num_layers = int(np.sum(np.array(dpn.DEFAULT_LAYERS_PER_BLOCK)))
        gates_per_layer = [1] * total_num_layers


    # Print model summary
    model.summary()
    
    
    # Set up callbacks depending on options                            
    init_lr_val = args.lr
    num_epochs = args.epochs
    save_weights = args.save_weights

    callbacks = [ cb_utils.CosineLRScheduler(init_lr_val, num_epochs) ]       

    # Save weights if requested
    if save_weights:
        weight_path = weight_dirname + "weights-cifar100-"
        weight_path += (str(final_dropout) + "-" + str(final_droppath) + 
                            "-" + str(init_lr_val) + "-" + str(num_epochs) + 
                            ".hdf5")
        
        print(weight_path)
        callbacks.append(ModelCheckpoint(weight_path, 
                            monitor="acc", period=int(num_epochs // 2),
                            save_best_only=False, save_weights_only=True))
                            
    # Set up increasing Dropout callback
    if final_dropout != 0:
        callbacks.append(cb_utils.DynamicDropoutWeights(final_dropout))
            
    # Set up increasing stochastic depth (aka "Drop Path") callback
    if final_droppath != 0:
        if model_type in DROPPATH_MODELS:
            callbacks.append(cb_utils.DynamicDropPathGates(
                                                final_droppath, drop_table, 
                                                gates_per_layer))
        else:
            raise ValueError("Droppath flag cannot be used with '%s'.\n" \
                                "It is only compatible with the following models:  %s" % 
                                (model_type, ", ".join(DROPPATH_MODELS)))
        
    
    # Compile model and conduct training
    batch_size = 64
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
