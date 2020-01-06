import numpy as np
import pandas as pd
import settings
import re
import string
import os

os.environ['KMP_WARNINGS'] = '0'

months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
          'november', 'december']
from keras.layers import LSTM, Activation, Dense, Dropout, TimeDistributed, Input, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import load_model
import tensorflow as tf
import joblib
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence


class NeuralNetwork(): #give this arguments like: model type, train/test file
    #max_words = 900
    #max_len = 15
    #y_dict = {}
    epochs = 15
    batch_size = 15
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model_path = settings.model_path

    def __init__(self, model_name='trans'):
        self.model_name = 'page_id_' + model_name
        self.model_loc = self.model_path + self.model_name + '.h5'
        self.tok_loc = self.model_path + self.model_name + 'tokeniser.joblib'

    def train(self, file=settings.dataset_path + 'identified_trans_marginals_dataset.csv'):
        df = pd.read_csv(file)
        self.X = df['transformed']
        self.Y = df['tag']
        self.max_words, self.max_len = check_maxlens(df)
        self.tok = Tokenizer(num_words=self.max_words)
        self.model = self.LSTM()

        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.15)

        self.tok.fit_on_texts(X_train)
        sequences = self.tok.texts_to_sequences(X_train)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len)
        y_binary = to_categorical(Y_train)
        self.model.summary()
        self.model.fit(sequences_matrix, y_binary, batch_size=self.batch_size, epochs=self.epochs,
                  validation_split=0.2) #, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]

        test_sequences = self.tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=self.max_len)

        accr = self.model.evaluate(test_sequences_matrix, to_categorical(Y_test))
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        self.model.save(self.model_loc)
        joblib.dump(self.tok, self.tok_loc)

    def LSTM(self):
        model = Sequential()
        model.add(Embedding(self.max_words, output_dim=self.max_len))#256))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model


    def load_model_from_file(self):
        self.model = load_model(self.model_loc)
        self.tok = joblib.load(self.tok_loc)
        self.model._make_predict_function()


    def predict(self, strings):
        if not os.path.exists(self.model_loc):
            self.train()
        try:
            self.model
        except AttributeError:
            self.load_model_from_file()

        strings = strings.apply(lambda x: transform_text(x))
        sequences = self.tok.texts_to_sequences(strings)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len)
        predictions = self.model.predict(sequences_matrix)
        return predictions, np.argmax(predictions, axis=1)


def check_maxlens(df):
    series = df['transformed']
    seqs = series.str.lower().str.split()
    max_seq_len = len(max(seqs, key=lambda x:len(x)))
    all_words = []
    seqs.apply(lambda x: all_words.extend(x))
    unique_words = len(set(all_words))
    #max_words = len(words.unique())
    return unique_words, max_seq_len


def transform_text(str):
    str = str.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    tokens = str.split(r' ')
    new_text = ''
    for token in tokens:
        if re.match(r'^\t', token):
            new_text += 'tab '
            token = token.strip('\t')
        if re.match(r'^[0-9][0-9]?$', token):
            new_text += 'smallNum '
        elif re.match(r'^[0-9]+$', token):
            new_text += 'bigNum '
        elif token.lower() in months:
            new_text += 'month '
        elif re.match(r'^Page$', token):
            new_text += 'page '
        elif re.match(r'^[a-zA-Z]+$', token):
            new_text += 'word '
        elif not re.match(r'^(|\s+)$', token):
            new_text += 'mix '
            print(token)
    return new_text


def create_dataset():
    sourcefile = settings.marginals_id_dataset
    texts = pd.read_csv(sourcefile, usecols=['Text'])
    new_text = texts.Text.apply(lambda x: transform_text(x))
    print(new_text)
    new_text.to_csv(settings.marginals_id_trans_dataset, index=False)


if __name__ == "__main__":
    #create_dataset():
    nn = NeuralNetwork()
    #nn.train()
    nn.load_model_from_file()
    p, r = nn.predict(pd.Series(['page 8', 'bhp hello world 3', '12 month report', 'epm3424 3 february 1900',
                                 'epm23 february 2000', 'epm34985 4000']))
    print(p)
    print('------------------')
    print(r)
