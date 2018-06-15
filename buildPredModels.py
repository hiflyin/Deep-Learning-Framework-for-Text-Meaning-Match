# -*- coding: utf-8 -*-
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential,optimizers
from keras.layers import LSTM, Activation, TimeDistributed, Dense, RepeatVector, Bidirectional, Dropout, RepeatVector
from keras.layers import Input, merge, Reshape, Flatten#, Conv1D,  Flatten
from keras.models import Model
from keras.optimizers import Nadam, SGD
from keras import regularizers
import pandas as pd
from gensim.models import word2vec
from timeit import default_timer as timer
from settings import *

callbacks = [EarlyStopping(monitor='val_loss', min_delta=DELTA_LOSS, patience=5, verbose=5, mode='auto'),
             ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=False, mode='auto', period=2)]

def show(message):
    print(" ")
    print(np.repeat("*", 50).tostring())
    print("   " + message)
    print(np.repeat("*", 50).tostring())

def generatePredictions(model):

    Q12_sents = np.append(np.load(TEST_Q1_SENTS), np.load(TEST_Q2_SENTS), axis=1)
    print(">>>>>>>> Total data size ... ")
    print(Q12_sents.shape)
    preds = model.predict(Q12_sents)
    np.save(PREDICTIONS_FILE, preds)

def getOptimizer():

    if OPTIMIZER == "Nadam":
        opt = optimizers.Nadam(lr=LEARNING_RATE, schedule_decay=Decay, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
    elif OPTIMIZER == "SGD":
        opt = optimizers.SGD(lr=LEARNING_RATE, decay=Decay, momentum=MOMENTUM, nesterov=NESTEROV)
    return opt


def trainModel3(X_train,  y_train, X_val, y_val):

    # create and fit the model
    print(">>>>>>>> Fitting model 3 ... ")
    print(">>>>>>>> Train data size is {} ... ".format(X_train.shape))
    print(">>>>>>>> Val data size is {} ... ".format(X_val.shape))
    hidden_size = WORD_ENC_SIZE + 2*max_sent_length
    layer1 = Bidirectional(LSTM(hidden_size, return_sequences=True),  input_shape=(X_train.shape[1], X_train.shape[2]), merge_mode='sum')
    #layer3 = Dense(1)
    model = Sequential()
    model.add(layer1)
    for i in range(N_REC_LAYERS):
        model.add(Bidirectional(LSTM(hidden_size, return_sequences=True),merge_mode='sum'))
    model.add(Bidirectional(LSTM(2*max_sent_length, return_sequences=False), merge_mode='sum'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(KERNEL_L2)))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=getOptimizer(), metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, nb_epoch=NB_EPOCH, batch_size=batch_size, verbose=2, shuffle=True, callbacks=callbacks,
              validation_data=[X_val, y_val], initial_epoch = 12)

    return model

def trainModel31(X_train, X_cosines_train, y_train, X_val, X_cosines_val, y_val):

    # create and fit the model
    print(">>>>>>>> Fitting model 31 ... ")
    print(">>>>>>>> Train data size is {} ... ".format(X_train.shape))
    print(">>>>>>>> Val data size is {} ... ".format(X_val.shape))
    hidden_size = WORD_ENC_SIZE + 2*max_sent_length
    input_X = Input(shape=(X_train.shape[1], X_train.shape[2]), name='input_X')
    input_cosines = Input( shape = (1,1), name='cosines')
    layer1 = Bidirectional(LSTM(hidden_size, return_sequences=True),  input_shape=(X_train.shape[1], X_train.shape[2]), merge_mode='sum')(input_X)
    layer2 = Bidirectional(LSTM(2*max_sent_length, return_sequences=False), merge_mode='sum')(layer1)
    layer3 = Dense(1, kernel_regularizer=regularizers.l2(.04), activation = "sigmoid")(layer2)
    layer4 = Flatten()(input_cosines)
    concat = merge([layer3, layer4], mode='concat', concat_axis=1)
    output_layer = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(.02))(concat)
    model = Model(input=[input_X, input_cosines], output=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=getOptimizer(), metrics=['accuracy'])
    print(model.summary())
    model.fit([X_train, X_cosines_train], y_train, nb_epoch=NB_EPOCH, batch_size=batch_size, verbose=2, shuffle=True, callbacks=callbacks,
              validation_data=[[X_val, X_cosines_val], y_val])

    return model


def generateValIds():

    y = pd.read_csv("data/Y.csv").y.values
    print(">>>>>>>> Length of train data is {}".format(len(y)))
    test_set = int(TEST_PERC * len(y))
    class1_perc = int(float(sum(y == 1)) / len(y) * test_set)
    class2_perc = test_set - class1_perc
    test_ids = []
    for i in range(10):
        cl1_ids = np.random.choice(np.where(y == 1)[0], size=class1_perc, replace=False, p=None)
        cl2_ids = np.random.choice(np.where(y == 0)[0], size=class2_perc, replace=False, p=None)
        test_ids.append(np.append(cl1_ids, cl2_ids))
    test_ids = np.array(test_ids)
    print(">>>>>>>> Generated test ids with the following shape {}".format(test_ids.shape))
    np.save(TEST_IDS, test_ids)

def train_val_model():

    Q12_sents = np.append(np.load(TRAIN_Q1_SENTS), np.load(TRAIN_Q2_SENTS), axis=1)
    y = pd.read_csv("data/Y.csv").y.values[0:Q12_sents.shape[0]]
    cosines = np.load("Train_cosines.npy")[0:Q12_sents.shape[0]]
    cosines = np.reshape(cosines, (cosines.shape[0], 1, 1))
    class1_size = int(len(np.where(y==1)[0].tolist()) * TEST_PERC)
    class2_size = int(len(np.where(y == 0)[0].tolist()) * TEST_PERC)
    val_ids = np.where(y==1)[0].tolist()[0:class1_size] + np.where(y==0)[0].tolist()[0:class2_size]
    model = trainModel31(np.delete(Q12_sents, val_ids, axis = 0), np.delete(cosines, val_ids, axis=0), np.delete(y, val_ids),
                         Q12_sents[val_ids], cosines[val_ids], y[val_ids])
    print("Model was trained on {}".format(Q12_sents.shape))
    model.save(FINAL_MODEL)


if __name__ == '__main__':

    start = timer()
    show("Generating test ids for cross-val ... ")
    generateValIds()
    print("Elapsed time: {}".format(timer()- start))
    start = timer()
    start = timer()
    show("Assesing models on crossval ... ")
    train_val_model(int(MODEL_ID))
    print("Elapsed time: {}".format(timer()- start))
    start = timer()


