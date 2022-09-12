# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:49:59 2022

@author: Arya
"""

# This file contains the functions for feature extraction related to the link : https://medium.com/@sdoshi579/classification-of-music-into-different-genres-using-keras-82ab5339efe0 : 
# libraries
from config import config
import numpy as np
import pandas as pd
import csv
import librosa


# return a list of audio Digital Signal plus the audio name
def audio_list( list_audio, wav_path ):
    list_audio_pd   = pd.read_csv( list_audio )
    
    for audio_name in list( list_audio_pd.Audio ):
        audio_path  = wav_path + audio_name + '.wav'
        y           = librosa.load( audio_path, sr = config['sr'] )
        
        yield y[0], audio_name
        
# feature extraction        
def feature_extraction_application_music_genre (  list_audio, wav_path, csv_file_features_to_save  ):
    
    header = 'Audio chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()
    
    sr     = config['sr']
    
    file   = open( csv_file_features_to_save, 'w', newline='' )
    with file:
        writer = csv.writer( file )
        writer.writerow( header )

    for y, audio_name in audio_list( list_audio, wav_path ):
        
        chroma_stft = librosa.feature.chroma_stft( y=y, sr=sr)
        rmse        = librosa.feature.rms(y=y)
        spec_cent   = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw     = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr         = librosa.feature.zero_crossing_rate(y)
        mfcc        = librosa.feature.mfcc(y=y, sr=sr)
        
        to_append   = f'{audio_name} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        
        file   = open( csv_file_features_to_save, 'a', newline='' )
        with file:
            writer  = csv.writer( file )
            writer.writerow( to_append.split() )
            
    return

