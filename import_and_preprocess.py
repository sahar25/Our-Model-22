# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:44:27 2022

@author: Arya
"""
# libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle

# standardize data and return the scaler for future test data
def standardize_train_test ( train_data, test_data ):
    
    X_train       = np.array( train_data )
    X_test        = np.array( test_data )
    
    scaler        = preprocessing.StandardScaler().fit( X_train )
    X_train_stand = scaler.transform( X_train )
    X_test_stand  = scaler.transform( X_test )
    
    return X_train_stand, X_test_stand, scaler

def save_model_info ( info, path ):
    pickle.dump ( info, open( path, 'wb' ))
    

# one hot encoding y train and y test
def y_preprocessing ( y_train_path, y_test_path, ohe_path ):
    
    loaded_encoder     = pickle.load( open(ohe_path, 'rb') )
    
    y_train_pd         = pd.read_csv( y_train_path )
    y_train            = np.array(y_train_pd.TARGET).reshape((-1, 1))
    y_train_oh_encoded = loaded_encoder.transform( y_train ).toarray()
    
    y_test_pd          = pd.read_csv( y_test_path )
    y_test             = np.array(y_test_pd.TARGET).reshape((-1, 1))
    y_test_oh_encoded  = loaded_encoder.transform( y_test ).toarray()
    
    return y_train_oh_encoded, y_test_oh_encoded


# preprocess and return train and test data for a DL model
def preprocess__ ( X_train_path, X_test_path, y_train_path, y_test_path, ohe_path, scaler_path ):
    
    train_data                            = pd.read_csv( X_train_path )
    test_data                             = pd.read_csv( X_test_path  )   
    X_train_stand, X_test_stand, scaler   = standardize_train_test ( train_data.iloc[:, 1:], test_data.iloc[:, 1:] )
    
    save_model_info ( scaler, scaler_path )
    
    y_train_oh_encoded, y_test_oh_encoded = y_preprocessing ( y_train_path, y_test_path, ohe_path )
    
    return X_train_stand, X_test_stand, y_train_oh_encoded, y_test_oh_encoded
    
    
    
    