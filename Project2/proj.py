
# coding: utf-8

# In[129]:


#!/usr/bin/env python
# coding: utf-8

import struct as st
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
from sklearn.model_selection import GridSearchCV

def read_data(images, labels):
    filename = {'images' : images ,'labels' : labels}
    imagesfile = open(filename['images'],'rb')

    imagesfile.seek(0)
    st.unpack('>4B',imagesfile.read(4))

    nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images
    nR = st.unpack('>I',imagesfile.read(4))[0] #num of rows
    nC = st.unpack('>I',imagesfile.read(4))[0] #num of column

    data = np.zeros((nImg,nR,nC))

    nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
    data = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC))

    return data

def read_labels(labels_file):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    labels = np.fromfile(labels_file ,dtype = 'ubyte' )[2 * intType.itemsize:]
    return labels

def create_model(learn_rate=0.01, activation='relu', dropout=0.2):
    # create model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(1,28,28),data_format="channels_first" ,activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    
    model.add(Conv2D(16, (3, 3),activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
        
    model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    opt = optimizers.adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# TODO: need to update to plot fixing one param and vary over the others
def plot_results(params_means, best):

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
        
    row = 0
    for key in best:
        x = []
        means = []
        costs = []
        
        if key == 'learn_rate':
            fix = 'fixed learn rate: ' + str(best[key])
            xlabel = 'batch_size'
            param_curve = 'batch_size'
        else:
            fix = 'fixed batch size: ' + str(best[key])
            xlabel = 'learn_rate'
            param_curve = 'learn_rate'

        for param_mean in params_means:
            if param_mean[0][key] == best[key]:
                x.append(param_mean[0][param_curve])
                means.append(param_mean[1])
                costs.append(param_mean[2])

        axes[row, 0].set_title(fix)
        axes[row, 0].plot(x, means)
        axes[row, 0].set_xlabel(xlabel)
        axes[row, 0].set_ylabel('Accuracy')
        #plt.sca(axes[row, 0])
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        #plt.xticks(x, rotation=75)
        
        
        axes[row, 1].set_title(fix)
        axes[row, 1].plot(x, costs)
        axes[row, 1].set_xlabel(xlabel)
        axes[row, 1].set_ylabel('Cost')
        #plt.sca(axes[row, 1])
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        #plt.xticks(x, rotation=75)
        
        row += 1

    fig.tight_layout()
    plt.show()
    plt.savefig('cv_plot.png', dp1=100)
    
    
def plot_activation_curves(relu_hist, sig_hist, tan_hist):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 9))

    # RELU
    # # for accuracy
    axes[0,0].plot(relu_hist.history['acc'])
    axes[0,0].plot(relu_hist.history['val_acc'])
    axes[0,0].set_title('relu ~ model accuracy')
    axes[0,0].set_ylabel('accuracy')
    axes[0,0].set_xlabel('epoch')
    axes[0,0].legend(['train', 'validation'], loc='upper left')
    # # summarize history for loss
    axes[0,1].plot(relu_hist.history['loss'])
    axes[0,1].plot(relu_hist.history['val_loss'])
    axes[0,1].set_title('relu ~ model loss')
    axes[0,1].set_ylabel('loss')
    axes[0,1].set_xlabel('epoch')
    axes[0,1].legend(['train', 'validation'], loc='upper left')
    
    # Sigmoid
    # # for accuracy
    axes[1,0].plot(sig_hist.history['acc'])
    axes[1,0].plot(sig_hist.history['val_acc'])
    axes[1,0].set_title('sigmoid ~ model accuracy')
    axes[1,0].set_ylabel('accuracy')
    axes[1,0].set_xlabel('epoch')
    axes[1,0].legend(['train', 'validation'], loc='upper left')
    # # summarize history for loss
    axes[1,1].plot(sig_hist.history['loss'])
    axes[1,1].plot(sig_hist.history['val_loss'])
    axes[1,1].set_title('sigmoid ~ model loss')
    axes[1,1].set_ylabel('loss')
    axes[1,1].set_xlabel('epoch')
    axes[1,1].legend(['train', 'validation'], loc='upper left')
    
    # Tanh
    # # for accuracy
    axes[2,0].plot(tan_hist.history['acc'])
    axes[2,0].plot(tan_hist.history['val_acc'])
    axes[2,0].set_title('tanh ~ model accuracy')
    axes[2,0].set_ylabel('accuracy')
    axes[2,0].set_xlabel('epoch')
    axes[2,0].legend(['train', 'validation'], loc='upper left')
    # # summarize history for loss
    axes[2,1].plot(tan_hist.history['loss'])
    axes[2,1].plot(tan_hist.history['val_loss'])
    axes[2,1].set_title('tanh ~ model loss')
    axes[2,1].set_ylabel('loss')
    axes[2,1].set_xlabel('epoch')
    axes[2,1].legend(['train', 'validation'], loc='upper left')

    fig.tight_layout()
    plt.show()
    plt.savefig('activations_plot.png', dp1=100)


# In[118]:



train_data = read_data('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
print (train_data.shape)

test_data = read_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
print (test_data.shape)

train_labels = read_labels('train-labels.idx1-ubyte')
test_labels = read_labels('t10k-labels.idx1-ubyte')

# normalize data:
train_data = train_data/255
test_data = test_data/255

# add channels to data
train_data = train_data.reshape(train_data.shape[0], 1, 28, 28).astype('float32')
test_data = test_data.reshape(test_data.shape[0], 1, 28, 28).astype('float32')

# shuffle and split training data
seed = 7
np.random.seed(seed)

train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.33, random_state=42)


# In[97]:


# PARAMETER SELECTION ~ using cross validation using grid search

# search for best parameter combination
model_check = KerasClassifier(build_fn=create_model, activation='relu', epochs=5, dropout=0.2, verbose=2)

learn_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
batch_size = [16,32,64,128,256]

param_grid = dict(learn_rate=learn_rate, batch_size=batch_size)
grid = GridSearchCV(estimator=model_check, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_val, y_val)


# In[98]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
best = grid_result.best_params_
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[104]:


plot_results(list(zip(params, means, stds)), best)


# In[113]:


## LEARNING CURVE FOR ACTIVATIONS
relu_model = create_model(activation='relu', learn_rate = best['learn_rate'])
relu_hist = relu_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=best['batch_size'], verbose=2)

sig_model = create_model(activation='sigmoid', learn_rate = best['learn_rate'])
sig_hist = sig_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=best['batch_size'], verbose=2)

tan_model = create_model(activation='tanh', learn_rate = best['learn_rate'])
tan_hist = tan_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=best['batch_size'], verbose=2)


# In[114]:


# PLOT LEARNING CURVES FOR ACTIVATIONS
plot_activation_curves(relu_hist, sig_hist, tan_hist)


# In[137]:


## FINAL MODEL to be trained with all data (training + validation)
final_model = create_model(activation='sigmoid', learn_rate = best['learn_rate'])
final_model.fit(train_data, train_labels, epochs=10, batch_size=best['batch_size'], verbose=2)
# final_model = create_model(activation='sigmoid', learn_rate = 0.01)
# final_model.fit(train_data, train_labels, epochs=10, batch_size=32, verbose=2)


# In[141]:


## PREDICTIONS (to csv file)
predictions = final_model.predict_classes(test_data)

predictions_matrix = np.full((10000, 10), 0)
for index in range(0,10000):
    predictions_matrix[index][predictions[index]] = 1
    
cnn_csv = np.asarray(predictions_matrix)
np.savetxt("mnist.csv", cnn_csv.astype(int), fmt='%i', delimiter=",")


# In[139]:


# SCORE
scores = final_model.evaluate(test_data, test_labels, verbose=0)
print("CNN Error: %.2f%%" % (scores[1]*100))

