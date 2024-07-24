#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bayo, luque, malagon
Goals: 
    class Args:  contains global parameters
    other functions: custom functions
Last modification: Jul 24, 2024
"""

import os
import numpy as np
import tensorflow as tf
import random
from matplotlib import pyplot as plt
from h5py import File
from sys import exit
import json
import tomllib


###############################################################################
#   C L A S S   A R G S
###############################################################################

class Args:
    """
    Class used to initialize global parameters useful for execution
    """
    
    def __init__(self, fname):
        """
        Initialize the global parameters and creates the folder to store the model
        
        Parameters
        ----------
        fname: the path of the file with the configuration parameters
        """    
        
        with open(fname, "rb") as fp:
            config = tomllib.load(fp)

        self.project_path = os.path.join(os.getcwd(), '') # Folder where the executable is located and the results will be stored
        self.model_path = os.path.join(self.project_path, "model/") # Folder to store the best trained model
        self.seed = config.get('seed', 2024)    # Initial SEED for reproducibility
        self.epochs = config.get('epochs', 1000) # The number of epochs for training
        self.patience = config.get('patience', 50) # For early_stopping callback (the number of iterations with no improvement)
        self.monitor = config.get('monitor', 'val_loss') # Metric used to reduce learning rate when there is no improvement
        self.noisy_files = os.path.join(self.project_path, config.get('noisy_files', 'data/noisy/*.hdf')) # Folder containing noisy files
        self.truth_files = os.path.join(self.project_path, config.get('truth_files', 'data/ground/*.hdf')) # Folder containing ground-truth files
        self.datasets_size = config.get('datasets_size', [0.7, 0.2, 0.1]) # The size (percentage) of the training, validation, and testing data sets. You must add 1
        self.batch_size = config.get('batch_size', 8)    # The batch size
        self.conv_num_filters = config.get('conv_num_filters', 4)   # The initial number of filters in convolutional operations
        self.conv_kernel_size = config.get('conv_kernel_size', 3)   # The size of the 2D kernel in convolutional operations
        
        os.makedirs(self.model_path, exist_ok=True)
        
        self.__dump_json()

        
    def __dump_json(self):
        """
        Write the configuration parameters to the "config.json" file
        
        Returns
        -------
        A file containing the parameters used
        """
        params_dict = {key: getattr(self, key) for key in self.__dict__}
        with open(os.path.join(self.project_path, "config.json"), "w") as fp:
            json.dump(params_dict, fp, indent = 4)   


###############################################################################
#   F U N C T I O N S
###############################################################################

def obtainDatasets(args):
    """  
    Data generator to feed the neural network training

    Returns
    -------
    The three data sets used to train, validate and test the algorithm: train_ds, val_ds and test_ds
    """

    def loadHdfFile(noisy_file):
    
        """
        Load the noisy streamer indicated by filename and its corresponding clean one
        
        Parameters
        ----------
        noisy_file: the name of the noisy streamer
    
        Returns
        -------
        Two arrays with the charge values of the noisy and the clean files
        """
    
        noisy_path = noisy_file.numpy().decode("utf-8") 
        clean_path = noisy_path.replace('/noisy', '/ground').replace('_noisy_50', '').replace('_noisy_10', '').replace('_noisy_5',  '')
        
        # noisy file   
        file = File(noisy_path, mode='r')   # loading the hdf5 file
        patch = list(file.keys())[0]        # name of the first group contained in the file
        group = file[patch]                 # first group
        dataset = list(group.keys())[0]     # name of the first dataset in the group (c-density)
        values_n = np.array(group[dataset]) # obtaining values
        file.close()   
        # apply radius to noisy values
        r = 0.5 + np.arange(values_n.shape[1])  
        values_n *= r
        values_n = np.reshape(values_n, (values_n.shape[0], values_n.shape[1], 1))

        # clean file
        file = File(clean_path, mode='r')   # loading the hdf5 file
        patch = list(file.keys())[0]        # name of the first group contained in the file
        group = file[patch]                 # first group
        dataset = list(group.keys())[0]     # name of the first dataset in the group (c-density)
        values_c = np.array(group[dataset]) # obtaining values
        file.close()
        # apply radius to clean values
        values_c *= r
        values_c = np.reshape(values_c, (values_c.shape[0], values_c.shape[1], 1))
        return values_n, values_c


    AUTOTUNE = tf.data.experimental.AUTOTUNE
    PREFETCH = AUTOTUNE 
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

    # TRAIN DATASET
    full_ds = tf.data.Dataset.list_files(args.noisy_files, seed = args.seed, shuffle = True)
    nons = len(list(full_ds))   # number ofnoisy streamers 
    train_streamers = int(nons * args.datasets_size[0])
    valid_streamers = int(nons * args.datasets_size[1])
    test_streamers = nons - (train_streamers + valid_streamers)
    train_ds = full_ds.take(train_streamers)
    train_ds = train_ds.map(lambda x: tf.py_function(loadHdfFile, [x], Tout = (tf.float32, tf.float32)), num_parallel_calls = AUTOTUNE, deterministic=True).batch(batch_size = args.batch_size, drop_remainder = False).prefetch(PREFETCH)
    train_ds = train_ds.with_options(options)
    
    # VALIDATION DATASET
    val_ds = None
    val_ds = full_ds.take(valid_streamers)
    val_ds = val_ds.map(lambda x: tf.py_function(loadHdfFile, [x], Tout = (tf.float32, tf.float32)), num_parallel_calls = AUTOTUNE, deterministic=True).batch(batch_size = args.batch_size, drop_remainder = False).prefetch(PREFETCH)
    val_ds = val_ds.with_options(options)
        
    # TEST DATASET
    test_ds = None
    test_ds = full_ds.take(test_streamers)
    test_ds = test_ds.map(lambda x: tf.py_function(loadHdfFile, [x], Tout = (tf.float32, tf.float32)), num_parallel_calls = AUTOTUNE, deterministic=True).batch(batch_size = args.batch_size, drop_remainder = False).prefetch(PREFETCH)
    test_ds = test_ds.with_options(options)

    return train_ds, val_ds, test_ds


def plotCurves(history, args):
    """
    Plot the loss curve for training and validation

    Parameters
    ----------
    history : history obtained in the neural network training process
    args: object of type Class Args containing the parameters necessary for execution

    Returns
    -------
    A file with the corresponding plot

    """    
    
    plt.ioff()
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'b',linewidth=3.0)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.plot(history.history['val_loss'],'r',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig(os.path.join(args.project_path, "training_loss_curves.png"))
    plt.close('all') 


def setEnvironment(seed):
    """
    Initialize random number generators and define some environment variables for the use of the GPU

    Parameters
    ----------
    seed: seed to initialize random generators
    """    
    
    tf.keras.backend.clear_session()
    
    # Set environment variables to use GPU. It can be done here or in the system environment variables
    try: 
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    except:
        print("GPU initialize error:")
        exit(0)

    # Initialize random number generators
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def trainNN(train, validation, args):
    """  
    Train the neural network and store the best model obtained.
    
    Parameters
    ----------
    train: train dataset
    validation: validation dataset
    args: object of type Class Args containing the parameters necessary for execution

    Returns
    -------
    The history of the training process.
    """

    INPUT_SIZE = (None, None, 1)    # the size of one streamer  
    model = unet(INPUT_SIZE, args)      
    model.compile(loss = 'MSE', optimizer = 'Adam')           
    model.summary()
    RLOP = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, min_lr = 1e-5, min_delta = 0, cooldown=1, verbose = 1)
    ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'min', verbose = 1, patience = args.patience)    
    MC = tf.keras.callbacks.ModelCheckpoint(args.model_path, monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only = True)
    CALLBACKS = (RLOP, ES, MC)
    STEPS_PER_EPOCH = train.cardinality().numpy()               # number of batches for train
    VAL_STEPS_PER_EPOCH = validation.cardinality().numpy()      # number of batches for validation
    WORKERS = int(os.cpu_count() / 2)                           # WORKERS = cpu_count() # Number of CPUs for parallel operations
    
    history = model.fit(train, epochs = args.epochs, steps_per_epoch = STEPS_PER_EPOCH, batch_size = args.batch_size, validation_data = validation, validation_steps = VAL_STEPS_PER_EPOCH, validation_batch_size = args.batch_size, verbose = 2, callbacks = CALLBACKS, use_multiprocessing = True, workers = WORKERS)
    
    return history


def unet(input_size, args):   
    """  
    Create a Unet neural network for denoising
    
    Parameters
    ----------
    input_size: the shape of the input data (the size of the streamer)

    Returns
    -------
    A Unet for denoising purposes
    """

    
    def add_unet_block(inputs, num_filters, ks, sc = None, t = 0):
        # t :  0 = contraction stage  1 = expansion stage
        d = tf.keras.layers.UpSampling2D(size=2) (inputs) if t == 1 else inputs    
        d = tf.keras.layers.Conv2D(num_filters, kernel_size = ks, padding = 'same') (d)
        if t == 0:
            d = tf.keras.layers.AveragePooling2D(pool_size=2) (d)
        d = tf.keras.layers.GroupNormalization(groups = num_filters) (d)
        d = tf.keras.layers.ReLU()(d)
        if t == 1: d = tf.keras.layers.Concatenate() ([sc, d])
        return d

    # inputs
    inputs = tf.keras.Input(shape = input_size)

    # Contracting path
    c1 = add_unet_block(inputs, num_filters = args.conv_num_filters * 1, ks = args.conv_kernel_size, sc = None, t = 0)
    c2 = add_unet_block(c1, num_filters = args.conv_num_filters * 2, ks = args.conv_kernel_size, sc = None, t = 0)
    c3 = add_unet_block(c2, num_filters = args.conv_num_filters * 4, ks = args.conv_kernel_size, sc = None, t = 0)
    c4 = add_unet_block(c3, num_filters = args.conv_num_filters * 8, ks = args.conv_kernel_size, sc = None, t = 0)
    x = add_unet_block(c4, num_filters = args.conv_num_filters * 16, ks = args.conv_kernel_size, sc = None, t = 0)    # botleneck
    
    # Expanding path    
    x = add_unet_block(x, num_filters = args.conv_num_filters * 8, ks = args.conv_kernel_size, sc = c4, t = 1)
    x = add_unet_block(x, num_filters = args.conv_num_filters * 4, ks = args.conv_kernel_size, sc = c3, t = 1)
    x = add_unet_block(x, num_filters = args.conv_num_filters * 2, ks = args.conv_kernel_size, sc = c2, t = 1)
    x = add_unet_block(x, num_filters = args.conv_num_filters * 1, ks = args.conv_kernel_size, sc = c1, t = 1)

    # outputs
    x = tf.keras.layers.UpSampling2D(size=2) (x)
    outputs = tf.keras.layers.Conv2D(1, kernel_size = 1, padding = 'same') (x)
    
    # Model definition
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model
