#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bayo, luque, malagon
Goal: UNET for denoising streamers
Last modification: Jul 24, 2024
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # put this before any tf call.
import argparse
import tensorflow as tf
import streamers_denoising_functions as sdf # Our functions library


def main():
    
    # obtaining parameters from configuration file
    parser = argparse.ArgumentParser(description='UNET model for denoising streamers')
    parser.add_argument('--input', "-i", type=str, help = 'File to read input parameters from')
    if parser.parse_args().input == None:
        print();    parser.print_usage();    print()
        parser.exit()
    args = sdf.Args(parser.parse_args().input)
    
    # initialize and configure environment
    sdf.setEnvironment(args.seed)

    # Obtaining datasets
    train, val, test = sdf.obtainDatasets(args)

    # Neural Network training using train and validation datasets
    history = sdf.trainNN(train, val, args)
    
    # Evaluate on test dataset
    best_model = tf.keras.models.load_model(args.model_path)
    evaluation = best_model.evaluate(test)
    print("Test loss:", evaluation)

    # Plotting loss curves
    sdf.plotCurves(history, args)

    
if __name__ == '__main__':
    main()
