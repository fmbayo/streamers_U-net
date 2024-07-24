#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bayo, luque, malagon
Last modification on Jul 24, 2024
Goal: denoise an input streamer with our Unet

Format: ./streamers_denoising_test.py model_folder noisy_streamer plot_orientation
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # put this before any tf call.
import sys
import numpy as np
from h5py import File
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

def plotStreamer(streamer, orientation = 'portrait'):
    """
    Store a file with the plot of the streamer

    Parameters
    ----------
    streamer : The streamer to be plotted
    """
    plt.ioff()

    if orientation == 'landscape':    
        fig, ax = plt.subplots(figsize=(12, 5), ncols=1)
        streamer = np.transpose(streamer)
    else:
        fig, ax = plt.subplots(figsize=(5, 12), ncols=1)
        
    fig.suptitle("Denoised Streamer", fontsize = 18, fontweight ="bold", y = 0.95)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('r (cm)')
    im = ax.imshow(streamer, cmap="bwr")
    plt.subplots_adjust(left = None, bottom = 0.075, right = None, top = 0.9, wspace = 0.1, hspace = 0.1)
    
    if orientation == "landscape":
        fig.colorbar(im, ax=ax, pad = 0.2, location = "bottom")
    else:
        fig.colorbar(im, ax=ax, pad = 0.2, location = "right")

    plt.savefig(os.path.join(os.getcwd(), "denoised_streamer.png"))
    plt.close('all') 
   

def main():

    if (len(sys.argv) != 4):
        print("\n\n./streamers_denoising_test.py model_folder noisy_streamer plot_orientation\n\n")
        sys.exit(0)       
    
    # loading the noisy streamer
    file = File(sys.argv[2] , mode='r') # loading the hdf5 file
    patch = list(file.keys())[0]        # name of the first group contained in the file
    group = file[patch]                 # first group
    dataset = list(group.keys())[0]     # name of the first dataset in the group (c-density)
    values_n = np.array(group[dataset]) # obtaining values
    file.close()   

    # apply radius to noisy values
    r = 0.5 + np.arange(values_n.shape[1])  
    values_n *= r
    SHAPE = values_n.shape
    values_n = np.reshape(values_n, (1, SHAPE[0], SHAPE[1], 1))
    
    # loading the model and predicting
    model = load_model(sys.argv[1])    # Loading the Model
    denoised = model.predict(values_n, verbose=0)     # denoising
    denoised = np.reshape(denoised, (SHAPE[0], SHAPE[1]))
    denoised = denoised / r
    
    # plot the denoised streamer
    plotStreamer(denoised, orientation = sys.argv[3])


if __name__ == '__main__':
    main()
