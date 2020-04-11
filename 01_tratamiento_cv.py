
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


kfold = KFold(n_splits=NUM_CV, shuffle=True, random_state=123456)


np.savetxt(SAVE + "train_np.csv", GR_train_array, delimiter=",")
np.savetxt(SAVE + "target_np.csv", GR_train_target, delimiter=",")
np.savetxt(SAVE + "test_np.csv", GR_test_array, delimiter=",")


def get_model():
    word_in = Input(shape=(MAX_LEN, 1))
    x = word_in
    main_lstm = Bidirectional(LSTM(units=30, return_sequences=True, recurrent_dropout=0.4))(x)
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.4))(main_lstm)
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.4))(main_lstm)
    out = TimeDistributed(Dense(N_TAGS, activation="softmax"))(main_lstm)
    model = Model(word_in, out)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=['categorical_accuracy'])
    return model


k = 0
for train_index, test_index in kfold.split(X=GR_train_array, y=GR_train_target_):
    k += 1

    with open(SAVE + 'test_index_' + str(k) + '.txt', 'w') as f:
        for item in list(test_index):
            f.write("%s\n" % item)

    GR_tr = GR_train_array[train_index,:]
    target_tr = GR_train_target_[train_index,:]
    GR_val = GR_train_array[test_index,:]
    target_val = GR_train_target_[test_index,:]

    model = get_model()

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    checkpointer = ModelCheckpoint(filepath=SAVE + 'model_' + str(k) + '.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(SAVE + 'log_' + str(k) + '.txt')
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=8, verbose=1, factor=0.5, min_lr=1e-8)

    model.fit(GR_tr.reshape(GR_tr.shape[0], GR_tr.shape[1], 1),
              target_tr,
              batch_size=BATCH_SIZE,
              epochs=MAX_EPOCH,
              validation_data=[GR_val.reshape(GR_val.shape[0], GR_val.shape[1], 1), target_val],
              verbose=2,
              callbacks=[early_stop, checkpointer, csv_logger, learning_rate_reduction],
              shuffle=True)

    del model
    K.clear_session()
    gc.collect()



    # pred_cv = model.predict(GR_val.reshape(GR_val.shape[0], GR_val.shape[1], 1))
    # pred_df = pd.DataFrame(pred_cv, index=train_index)
    # pred_df.to_csv(SAVE + 'pred_cv_' + str(k) + '.csv')
    # target_df = pd.DataFrame(GR_train_target[test_index,:], index=train_index)
    # target_df.to_csv(SAVE + 'target_cv_' + str(k) + '.csv')


