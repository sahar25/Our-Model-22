# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:52:36 2022

@author: Arya
"""

#--------------------------------------------------------------libraries----------------------------------------------------------------------------#
from paths import paths
# from feature_extraction import feature_extraction_application_music_genre
from import_and_preprocess import preprocess__
from model_based_on_genre_classification_application import evaluate_model

#----------------------------------------------------------- Feature Extraction --------------------------------------------------------------------#
# #--- Train Data
# feature_extraction_application_music_genre( paths['X_train'] , paths['set1_3s_folder'] , paths['features_train'] )

# #--- Test Data
# feature_extraction_application_music_genre( paths['X_test'] , paths['set1_3s_folder'] ,  paths['features_test'] )

#-------------------------------------------------------------- Preprocessing -----------------------------------------------------------------------#

X_train_stand, X_test_stand, y_train_oh_encoded, y_test_oh_encoded = preprocess__ ( paths['features_train'], paths['features_test'], paths['y_train'], paths['y_test'], paths['ohe'], paths['scaler'] )

#------------------------------------------------------------------ Model ---------------------------------------------------------------------------#

evaluate_model ( X_train_stand, y_train_oh_encoded, X_test_stand, y_test_oh_encoded, paths['ohe'] )
