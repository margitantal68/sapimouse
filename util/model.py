import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt  

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn import metrics

from util.utils import create_userids
from util.fcn import build_fcn

import util.settings as stt

###########################################################
# EZ ITT A SZENTIRAS
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
###########################################################

def plot_training(history, model_name, metrics ='loss'):
    # list all data in history
    print(history.history.keys())
    keys = list(history.history.keys())
    plt.figure()
    if( metrics == 'loss'):
        plt.plot(history.history[keys[0]])
        plt.plot(history.history[keys[2]])
        plt.title('Model loss ' + model_name)
        plt.ylabel('loss')
    
    if( metrics == 'accuracy'):
        plt.plot(history.history[keys[1]])
        plt.plot(history.history[keys[3]])
        # plt.plot(history.history['categorical_accuracy'])
        # plt.plot(history.history['val_categorical_accuracy'])
        plt.title('Model accuracy '+model_name)
        plt.ylabel('accuracy')
    

    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()
    # plt.savefig(stt.TRAINING_CURVES_PATH+'/'+model_name+'_'+metrics+'.png', format='png')
    # plt.savefig(stt.TRAINING_CURVES_PATH+'/'+model_name+'_'+metrics+'.png')


# fcn_filters is used only for FCN model
# representation_learning = False --> split the dataset into train -validation -test
# representation_learning = True  --> split the dataset into train -validation, therefore we use more data for training and validation
# 
def train_model( df, model_name = "foo.h5", fcn_filters=128, representation_learning=False):
    userids = create_userids( df )
    # print(userids)
    nbclasses = len(userids)
    print('number of classes: '+ str(nbclasses) )
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)

    if (representation_learning == False ):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=stt.RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=stt.RANDOM_STATE)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=stt.RANDOM_STATE)

    print("Train, validation (and test shapes): ")
    print(X_train.shape)
    print(X_val.shape)
    if( representation_learning == False):
        print(X_test.shape)

    mini_batch_size = int(min(X_train.shape[0]/10, stt.BATCH_SIZE))
    if( model_name == "foo.h5"):
        model_name = stt.MODEL_NAME
    filepath = stt.TRAINED_MODELS_PATH + "/" + model_name

    print(filepath)
    cb, model = build_fcn((stt.FEATURES, stt.DIMENSIONS ), nbclasses, filepath, fcn_filters )
    # model.summary()

    X_train = np.asarray(X_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)

    # convert to tensorflow dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    BATCH_SIZE = mini_batch_size
    SHUFFLE_BUFFER_SIZE = 100

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    start_time = time.time()
    hist = model.fit(train_ds, 
                      epochs=stt.EPOCHS,
                      verbose=True, 
                      validation_data=val_ds, 
                      callbacks=cb)
    
    hist_df = pd.DataFrame(hist.history) 
  
    # plot training curve
    plot_training(hist, model_name, metrics ='loss')
    plot_training(hist, model_name, metrics ='accuracy')

    duration = time.time() - start_time
    print("Training duration: "+str(duration/60))
    
    if (representation_learning == False):
        X_test = np.asarray(X_test).astype(np.float32)
        y_true = np.argmax( y_test, axis=1)
        y_pred = np.argmax( model.predict(X_test), axis=1)
        accuracy = metrics.accuracy_score(y_true, y_pred)     
        print("Test accuracy: "+str(accuracy))
    return model

# Evaluate model on a dataframe
# df: dataframe
def evaluate_model( df ):
    print("Evaluate model: ")
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]

    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)
    # evaluate model   
    model = tf.keras.models.load_model(stt.TRAINED_MODELS_PATH + "/" + stt.MODEL_NAME)
    # model.summary()
    y_true = np.argmax( y, axis=1)
    X = np.asarray(X).astype(np.float32)
    y_pred = np.argmax( model.predict(X), axis=1)
    accuracy = metrics.accuracy_score(y_true, y_pred)     
    print(accuracy)



# Use a pretrained model for feature extraction
# Load the model, pop the last layer
def get_model_output_features( df, model_name ):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)
    model_path = stt.TRAINED_MODELS_PATH + '/' + model_name
    model = tf.keras.models.load_model(model_path)
    # model.summary()
    # print(model_name)
    feat_model = tf.keras.Sequential()
    for layer in model.layers[:-1]:
        feat_model.add(layer)
    # feat_model.summary()
    X = np.asarray(X).astype(np.float32)
    features = feat_model.predict(X)   
    df = pd.DataFrame( features )
    df['user'] = y 
    # df.to_csv('features.csv', header = False, index=False)  
    return df




