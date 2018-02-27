from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import pre
import w2v
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from numpy.random import seed
seed(1)


def train(src,dest,pivot_num,pivot_min_st,dim):
    outputs = pivot_num
    HUs =dim
    #get the representation learning training and vlidation data
    x, y, x_valid, y_valid,inputs= pre.preproc(pivot_num,pivot_min_st,src,dest)
    #train word2vec for the pivots embeddings 
    w2v.wo2ve(src, dest, pivot_num, pivot_min_st, HUs)
    filename = src + "_to_" + dest + "/" + "pivot_mat/" + "pivot_mat_" + src + "_" + dest + "_" + str(
        pivot_num) + "_" + str(pivot_min_st) + "_" + str(dim)
    mat = np.load(filename)



    model = Sequential()
    #make sure the pivot matrix (the decoder) is fixed
    frozen_layer = Dense(outputs, trainable=False)
    model.add(Dense(HUs,input_shape=(inputs,),init='glorot_normal'))
    model.add(Activation('sigmoid'))
    model.add(frozen_layer)
    model.add(Activation('sigmoid'))
    print model.summary()
    w=model.get_weights()
    #initialzing the decoder with the pivot embeddings
    w[2] =  mat.transpose()
    sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9)
    model.set_weights(w)

    model.compile(optimizer=sgd, loss='binary_crossentropy')

    #stops as soon as the validaion loss stops decreasing
    earlyStopping = EarlyStopping(monitor='val_loss', patience=0, mode='min')
    #saveing only the best model 
    save_best = ModelCheckpoint("best_model", monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    h=model.fit(x, y, batch_size=1,callbacks=[earlyStopping],nb_epoch=10,validation_data=(x_valid,y_valid), shuffle=True)
    print (h.history['val_loss'])[-1]
    weight_str = src + "_to_" + dest + "/weights/w_"+src+"_"+dest+"_"+str(pivot_num)+"_"+str(pivot_min_st)+"_"+str(HUs)
    filename = weight_str
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    #saving the entire model
    np.save(weight_str, model.get_weights())

