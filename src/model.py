"""
GRU 모델을 사용한다.
"""
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import LeakyReLU, TimeDistributed, GRU, Bidirectional
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.layers.merge import concatenate
from keras.constraints import max_norm
from keras.optimizers import Adam
import keras.backend as K
import numpy as np


INPUTSHAPE = (256, 4,)


def custom_acc(y_true, y_pred):
    # accuracy 함수
    return K.mean(K.round(y_true) == K.round(y_pred))


def np_custom_acc(y_true, y_pred):
    # accuracy 함수 (numpy 버전)
    return np.mean(np.round(y_true) == np.round(y_pred))


def time_distributed_layer(pool):
    # time distributed dense
    pool = Dropout(0.3)(pool)
    pool = TimeDistributed(Dense(512, kernel_constraint=max_norm(5.)))(pool)
    return pool


def layer(units, inter):
    # fully connected layer (BN, leaky-relu 사용)
    inter = Dense(units, kernel_constraint=max_norm(5.))(inter)
    inter = BatchNormalization()(inter)
    inter = LeakyReLU()(inter)
    inter = Dropout(0.2)(inter)
    return inter


def build_model():
    # 모델을 반환한다. (v4.2.2)

    inputs1 = Input(shape=INPUTSHAPE)
    # GRU
    inter = Bidirectional(GRU(512, return_sequences=True), merge_mode='concat')(inputs1)
    # pooling
    avg_pool = AveragePooling1D(pool_size=3)(inter)
    avg_pool = time_distributed_layer(avg_pool)
    max_pool = MaxPooling1D(pool_size=3)(inter)
    max_pool = time_distributed_layer(max_pool)
    inter = concatenate([avg_pool, max_pool])
    # fully connected layers
    inter = layer(1024, inter)
    inter = layer(256, inter)
    inter = layer(64, inter)
    inter = Flatten()(inter)
    inter = layer(1024, inter)
    inter = Dense(64, kernel_constraint=max_norm(5.))(inter)
    inter = LeakyReLU()(inter)
    outputs = Dense(2, activation='softmax')(inter)

    model = Model(inputs=inputs1, outputs=outputs)

    optimizer = Adam(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[custom_acc])

    return model
