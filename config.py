# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:52:03 2022

@author: Arya
"""

config = { 
           'sr'            : 44100,
           'activation'    : 'relu',
           'optimizer'     : 'adam',
           'loss'          : 'categorical_crossentropy',
           'metrics'       : ['accuracy'],
           'epochs'        : 20,
           'batch_size'    : 32,
           'verbose'       : False
           
           }