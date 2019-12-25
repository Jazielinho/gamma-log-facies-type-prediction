
import pandas as pd
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D, BatchNormalization
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
import gc

# DEFINE PARAMETERS
PATH = 'D:/Concursos_Kaggle/33_Gamma Log Facies Type Prediction/01_datos/'
TRAIN = 'CAX_LogFacies_Train_File.csv'
TEST = 'CAX_LogFacies_Test_File.csv'
SAVE = 'D:/Concursos_Kaggle/33_Gamma Log Facies Type Prediction/10_resultados/01_tratamiento_cv/'

# PARAMETERS
MAX_LEN = 1100
N_TAGS = 5
NUM_CV = 5
BATCH_SIZE = 512
MAX_EPOCH = 1000

# LOAD DATA
data_train = pd.read_csv(PATH + TRAIN)
data_test = pd.read_csv(PATH + TEST)

# CREATE INPUT AND OUTPUT
GR_train_data = data_train.groupby('well_id')['GR'].agg(lambda x: list(x))
GR_train_target = data_train.groupby('well_id')['label'].agg(lambda x: list(x))

GR_test_data = data_test.groupby('well_id')['GR'].agg(lambda x: list(x))

GR_train_array = np.array(GR_train_data.to_list())
GR_train_target = np.array(GR_train_target.to_list())

GR_test_array = np.array(GR_test_data.to_list())

GR_train_target_ = to_categorical(GR_train_target, num_classes=N_TAGS)




# LOAD MODEL
from keras.models import load_model

best_model_1 = load_model(SAVE + 'model_1.hdf5')
pred_test_1 = best_model_1.predict(GR_test_array.reshape(GR_test_array.shape[0], GR_test_array.shape[1], 1), batch_size=4096, verbose=1)
del best_model_1
K.clear_session()
gc.collect()

best_model_2 = load_model(SAVE + 'model_2.hdf5')
pred_test_2 = best_model_2.predict(GR_test_array.reshape(GR_test_array.shape[0], GR_test_array.shape[1], 1), batch_size=4096, verbose=1)
del best_model_2
K.clear_session()
gc.collect()

best_model_3 = load_model(SAVE + 'model_3.hdf5')
pred_test_3 = best_model_3.predict(GR_test_array.reshape(GR_test_array.shape[0], GR_test_array.shape[1], 1), batch_size=4096, verbose=1)
del best_model_3
K.clear_session()
gc.collect()

best_model_4 = load_model(SAVE + 'model_4.hdf5')
pred_test_4 = best_model_4.predict(GR_test_array.reshape(GR_test_array.shape[0], GR_test_array.shape[1], 1), batch_size=4096, verbose=1)
del best_model_4
K.clear_session()
gc.collect()

best_model_5 = load_model(SAVE + 'model_5.hdf5')
pred_test_5 = best_model_5.predict(GR_test_array.reshape(GR_test_array.shape[0], GR_test_array.shape[1], 1), batch_size=4096, verbose=1)
del best_model_5
K.clear_session()
gc.collect()





pred_test = (pred_test_1 + pred_test_2 + pred_test_3 + pred_test_4 + pred_test_5) / 5

pred_test_class = np.argmax(pred_test, axis=-1)

pred_test_list = pred_test_class.tolist()

pred_test_subir = []

for list_ in pred_test_list:
    pred_test_subir = pred_test_subir + list_


# UPLOAD PREDICTIONS
data_subir = pd.DataFrame({'unique_id': data_test.unique_id,
                           'label': pred_test_subir})

data_subir.to_csv('D:/Concursos_Kaggle/33_Gamma Log Facies Type Prediction/ten_sub.csv', index=False)