from tensorflow.keras.layers import Embedding, Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPool1D

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


### LSTM, Bi-LSTM, GRU, Conv1D 모델을 학습 시 호출하기 위해 클래스화함.

class LSTM_model() :
    def __init__(self, vocab_size, X_train, X_test, Y_train, Y_test, korean_package, embedding_dim = 100, hidden_units = 128):

        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_dim))
        self.model.add(LSTM(hidden_units))
        # self.model.add(Dense(3, activation='softmax'))
        self.model.add(Dense(3, activation='softmax'))

        model_save_path = './saved_model'
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint(f'{model_save_path}/{korean_package}_LSTM_best_model.h5', monitor='val_acc', mode='max',
                             verbose=1, save_best_only=True)


        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        # self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])


        history = self.model.fit(X_train, Y_train, epochs=15, callbacks=[es, mc],
                                 batch_size=64, validation_data=(X_test,Y_test), shuffle=True)



class GRU_model() :
    def __init__(self, vocab_size, X_train, X_test, Y_train, Y_test, korean_package, embedding_dim = 100, hidden_units = 128):
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_dim))
        self.model.add(GRU(hidden_units))
        self.model.add(Dense(3, activation='softmax'))

        model_save_path = './saved_model'
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint(f'{model_save_path}/{korean_package}_GRU_best_model.h5', monitor='val_acc', mode='max',
                             verbose=1, save_best_only=True)

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        history = self.model.fit(X_train, Y_train, epochs=15, callbacks=[es, mc],
                                 batch_size=64, validation_data=(X_test,Y_test), shuffle=True)


class BiLSTM_model() :
    def __init__(self, vocab_size, X_train, X_test, Y_train, Y_test, korean_package, embedding_dim = 100, hidden_units = 128):
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_dim))
        self.model.add(Bidirectional(LSTM(hidden_units)))
        self.model.add(Dense(3, activation='softmax'))

        model_save_path = './saved_model'
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint(f'{model_save_path}/{korean_package}_Bi_LSTM_best_model.h5', monitor='val_acc', mode='max',
                             verbose=1, save_best_only=True)

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        history = self.model.fit(X_train, Y_train, epochs=15, callbacks=[es, mc],
                                 batch_size=64, validation_data=(X_test,Y_test), shuffle=True)

class onedCNN_model() :
    def __init__(self, vocab_size, X_train, X_test, Y_train, Y_test, korean_package, embedding_dim = 300, hidden_units = 128,
                 dropout_ratio = 0.3, num_filters=256, kernerl_size=3):
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_dim))
        self.model.add(Dropout(dropout_ratio))
        self.model.add(Conv1D(num_filters, kernerl_size, padding='valid', activation='relu'))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(hidden_units, activation='relu'))
        self.model.add(Dropout(dropout_ratio))
        self.model.add(Dense(3, activation='softmax'))

        model_save_path = './saved_model'
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint(f'{model_save_path}/{korean_package}_onedCNN_best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        history = self.model.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_data=(X_test,Y_test), shuffle=True)
