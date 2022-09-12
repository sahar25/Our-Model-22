# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:26:06 2022

@author: Arya
"""

# libraries
import numpy as np
from tensorflow.keras.models import Sequential
from config import config
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle

def model_genre_app ( X_shape ):
    
    model = Sequential()
    model.add( Dense( 256, activation = config['activation'], input_shape = (X_shape, ) ) )
    model.add( Dense( 128, activation = config['activation'] ) )
    model.add( Dense( 64, activation = config['activation'] ) )
    model.add( Dense( 11, activation = 'softmax' ) )
    
    model.compile( optimizer = config['optimizer'], loss = config['loss'], metrics = config['metrics'] )
    
    return model


def train_model ( X_train_stand, X_test_stand, y_train_oh_encoded, y_test_oh_encoded ):
    
    model     = model_genre_app ( X_train_stand.shape[1] )
    history   = model.fit( X_train_stand, y_train_oh_encoded, epochs = config['epochs'], batch_size = config['batch_size'], validation_data = ( X_test_stand, y_test_oh_encoded ), verbose = config['verbose'] )
    model.evaluate( X_test_stand, y_test_oh_encoded )
    
    y_predict = model.predict( X_test_stand )
    
    return model, history, y_predict
    
    
    
def plot( history ):
    
    # accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

def accuracy(confusion_matrix):
    
    classes   = len(confusion_matrix)
    sum_cm    = sum(sum(confusion_matrix))
    list_accs = []
    for i in range(classes):
        acc   = 1.*(sum_cm + 2*confusion_matrix[i,i] - sum(confusion_matrix[i,:]) - sum(confusion_matrix[:,i]))/sum_cm
        list_accs.append(acc)
        
    return sum(list_accs)/float(classes)


def from_probabilities_to_one_hot_encoding ( y_probabilities ):
    
    n_classes                 = len(y_probabilities[0])
    y_res                     = np.zeros ( y_probabilities.shape )
    for i, max_index in enumerate(np.argmax( y_probabilities, axis = 1 )):
        y_res[ i, max_index ] = 1
    
    return y_res


def one_hot_encoding_to_names ( ohe_path, y_ohe ):
    
    loaded_encoder = pickle.load( open(ohe_path, 'rb') )
    return loaded_encoder.inverse_transform( y_ohe ).reshape( (-1,) )


def model_performance_report ( y_true_ohe, y_pred_prob, ohe_path ):
    
    y_pred_ohe           = from_probabilities_to_one_hot_encoding ( y_pred_prob )
    y_pred_classes_names = one_hot_encoding_to_names ( ohe_path, y_pred_ohe )
    y_true_classes_names = one_hot_encoding_to_names ( ohe_path, y_true_ohe )
    cm                   = confusion_matrix ( y_true_classes_names, y_pred_classes_names )
    
    print ( "--- model preformance report ----" )
    print ( "#--- Area Under Curve score ----#" )
    print ( " -- macro -- " )
    print ( roc_auc_score ( y_true_ohe, y_pred_prob ) )
    print ( " -- micro -- " )
    print ( roc_auc_score ( y_true_ohe, y_pred_prob, average = 'micro' ) )
    print ( "#--- Accuracy ------------------#" )
    print ( accuracy( cm ) )
    print ( "#--- Classification Report -----#" )
    print ( classification_report ( y_true_classes_names, y_pred_classes_names ) )
    print ( "#--- Model Summary -------------#" )
    print ( model.summary() )
    

def evaluate_model ( X_train_stand, y_train_oh_encoded, X_test_stand, y_test_oh_encoded, ohe_path ):
    
    model, history, y_predict  =  train_model ( X_train_stand, X_test_stand, y_train_oh_encoded, y_test_oh_encoded )
    plot( history )
    model_performance_report ( y_test_oh_encoded, y_predict, ohe_path )