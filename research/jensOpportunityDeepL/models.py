# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:41:56 2020

@author: Jieyun Hu
"""

'''
Apply different deep learning models on PAMAP2 dataset.
ANN,CNN and RNN were applied.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from sklearn import metrics
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, SimpleRNN, GRU, LSTM, GlobalMaxPooling1D,GlobalMaxPooling2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from datetime import datetime
import os


class models():
    def __init__(self, path):
        self.path = path
       
    
    def read_h5(self):
        f = h5py.File(path, 'r')
        X = f.get('inputs')
        y = f.get('labels') 
        #print(type(X))
        #print(type(y))
        self.X = np.array(X)
        self.y = np.array(y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.4, random_state = 1)
    
        print("X = ", self.X.shape)
        print("y =",self.y.shape)
        print(set(self.y))
        #return X,y
    
    def cnn_model(self, n_epochs = 50):
       # K = len(set(y_train))
        #print(K)
        K = len(set(self.y))
        #X = np.expand_dims(X, -1)
        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test,-1)
        #print(X)
        #print(X[0].shape)
        #i = Input(shape=X[0].shape)
        i = Input(shape=self.x_train[0].shape)
        x = Conv2D(32, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(i)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(64, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Conv2D(128, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)    
        x = Dropout(0.2)(x)
        x = Dense(1024,activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(K, activation = 'softmax')(x)       
        self.model = Model(i,x)
        self.model.compile(optimizer = Adam(lr=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

        #self.r = model.fit(X, y, validation_split = 0.4, epochs = 50, batch_size = 32 )
        self.r = self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs = n_epochs, batch_size = 32 )
        print(self.model.summary())
        # It is better than using keras do the splitting!!
        return self.r
    
    def dnn_model(self):
       # K = len(set(y_train))
        #print(K)
        K = len(set(self.y))
        print(self.x_train[0].shape)
        i = Input(shape=self.x_train[0].shape)
        x = Flatten()(i)
        x = Dense(128,activation = 'relu')(x)
        x = Dense(128,activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(256,activation = 'relu')(x)
        x = Dense(256,activation = 'relu')(x)
        x = Dense(256,activation = 'relu')(x)
        #x = Dropout(0.2)(x)
        x = Dense(1024,activation = 'relu')(x)
        x = Dense(K,activation = 'softmax')(x)
        self.model = Model(i,x)      
        self.model.compile(optimizer = Adam(lr=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
        
        '''
        K = len(set(self.y))
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=self.x_train[0].shape),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(K,activation = 'softmax')
        ])
        model.compile(optimizer = Adam(lr=0.0005),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
        '''
        self.r = self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs = 50 )
        print(self.model.summary())
        return self.r
    

    def rnn_model(self):
        K = len(set(self.y))
        i = Input(shape = self.x_train[0].shape)
        x = LSTM(256, return_sequences=True)(i)
        x = Dense(128,activation = 'relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(K,activation = 'softmax')(x)
        self.model = Model(i,x)      
        self.model.compile(optimizer = Adam(lr=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
        self.r = self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs = 50, batch_size = 32 )
        #self.r = model.fit(X, y, validation_split = 0.2, epochs = 10, batch_size = 32 )
        print(self.model.summary())
        return self.r
   
    def draw(self, path_to_model_folder):
        f1 = plt.figure(1)
        plt.title('Loss')
        plt.plot(self.r.history['loss'], label = 'loss')
        plt.plot(self.r.history['val_loss'], label = 'val_loss')
        plt.legend()
        f1.show()

        # new: doesnt work (also not with plt) - no line in the chart
        # f1.savefig(path_to_model_folder + '/training_loss.png')
        
        f2 = plt.figure(2)
        plt.plot(self.r.history['accuracy'], label = 'accuracy')
        plt.plot(self.r.history['val_accuracy'], label = 'val_accuracy')
        plt.legend()
        f2.show()

        # new: doesnt work (also not with plt) - no line in the chart
        # f2.savefig(path_to_model_folder + '/training_acc.png')
        
    # summary, confusion matrix and heatmap
    def evaluation(self, model_folder_path):
        K = len(set(self.y_train))
        self.y_pred = self.model.predict(self.x_test).argmax(axis=1)

        # accuracy
        accuracy = np.sum(self.y_pred == self.y_test) / len(self.y_pred)
        print('===> Accuracy on Test: %f' % accuracy)
        with open(model_folder_path + '/evaluation.txt', 'w') as f: 
            f.write('Accuracy on Test: ' + str(accuracy))

        # confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        self.plot_confusion_matrix(cm, list(range(K)), model_folder_path) # jens plot function, extended

            
    
    def plot_confusion_matrix(self, cm, classes, path_to_model_folder, normalize = False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")
        print(cm)
        f3 = plt.figure(3)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # this was the code: invalid syntax: fmt = '.2f' if normalize else 'd' tra
        fmt = '.2f' if normalize else 'd'

        thresh = cm.max()/2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment = "center",
                     color = "white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('predicted label')
            f3.show()
        
        # new: also save
        plt.savefig(path_to_model_folder + '/conf_matrix.png')
    
    def get_model_name(self) -> str:
        if self.model_name is None:
            currentDT = datetime.now()
            currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
            self.model_name = currentDT_str + "---" + type(self).__name__
        return self.model_name

    def save_model(self, current_path_in_repo, model_name) -> None:
        """
        Saves the model to the given path

        returns the model_folder_path
        """
        # create directory
        currentDT = datetime.now()
        currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")

        model_name = currentDT_str + '-' + model_name

        path_saved_models = current_path_in_repo + '/saved_models/'
        model_folder_path = os.path.join(path_saved_models, model_name)
        model_folder_path_internal = os.path.join(model_folder_path, "model")
        os.makedirs(model_folder_path_internal, exist_ok=True)

        # save normal model
        self.model.save(model_folder_path_internal)

        # save model as .h5
        self.model.save(model_folder_path + "/" + model_name + ".h5", save_format='h5')

        # save model as .tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # TODO: Optimizations for new tensorflow version
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

        tflite_model = converter.convert()
        # converter = tf.lite.TFLiteConverter.from_saved_model(model_folder_path_internal)
        # tflite_model = converter.convert()
        print(f"Saving TFLite to {model_folder_path}/{model_name}.tflite")
        with open(f"{model_folder_path}/{model_name}.tflite", 'wb') as f:
            f.write(tflite_model)
        
        #  return model_folder_path + "/" + model_name + ".h5", f"{model_folder_path}/{model_name}.tflite"
        return model_folder_path


if __name__ == "__main__":
    model_name = "cnn" # can be cnn/dnn/rnn
    loco = False # True is to use locomotion as labels. False is to use high level activities as labels

    current_path_in_repo = 'research/jensOpportunityDeepL'
    path = current_path_in_repo #  keep in mind to change it in dataProcessing.py as well 
    # (current_path_in_repo = 'research/jensOpportunityDeepL')

    if loco:
        path += "/loco_2.h5"
    else:
        path += "/hl_2.h5"
        
    oppo = models(path) # only path
    
    print("read h5 file....")
    oppo.read_h5() # read, test train split
    '''
        oppo.X.shape # (34181, 25, 220) -> 34181 windows, 25 timestamps (under 1 sec), 220 features (sensor values)
        oppo.y.shape # (34181,) -> 34181 labels
        oppo.x_train.shape # (20508, 25, 220)
        oppo.y_train.shape # (20508,)
    '''

    if model_name == "cnn":
        oppo.cnn_model(n_epochs = 10) # n_epochs = 1 for testing, jens recommends 50
    elif model_name == "dnn":
        oppo.dnn_model()
    elif model_name == "rnn":
        oppo.rnn_model()

    # new
    model_folder_path = oppo.save_model(current_path_in_repo, model_name)

    oppo.draw(model_folder_path) # todo: no line visible at the moment

    oppo.evaluation(model_folder_path) # plots acc and confusion matrix

    

