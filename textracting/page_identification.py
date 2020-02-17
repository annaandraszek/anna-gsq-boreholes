import numpy as np
import pandas as pd
import settings
import re
import string
import active_learning
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing import sequence
from keras.models import load_model
import tensorflow as tf
import joblib
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
import machine_learning_helper as mlh
from sklearn.pipeline import Pipeline
from heading_id_toc import Text2Seq

os.environ['KMP_WARNINGS'] = '0'

months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
          'november', 'december']
name = 'page_id'
y_column = 'tag'
limit_cols = ['original']


class NeuralNetwork():
    epochs = 15
    batch_size = 15
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    #model_path = settings.model_path

    def __init__(self, model_name='trans'):
        #self.model_name = 'page_id_' + model_name
        self.model_loc = settings.get_model_path('page_id') #self.model_path + self.model_name + '.h5'
        self.tok_loc = settings.get_model_path('page_id', tokeniser=True)  #self.model_path + self.model_name + 'tokeniser.joblib'

    def train(self, file=settings.get_dataset_path('page_id'), n_queries=10):  #settings.marginals_id_trans_dataset):
        df = pd.read_csv(file)
        #self.X = df['transformed']
        #self.Y = df['tag']
        self.max_words, self.max_len = check_maxlens(df)

        lstm = KerasClassifier(build_fn=self.LSTM, batch_size=self.batch_size, epochs=self.epochs,
                               validation_split=0.2)

        estimator = Pipeline([
            ('transform', Text2Seq(classes=2)),
            ('lstm', lstm)
        ], verbose=True)

        accuracy, learner = active_learning.train(df, y_column, n_queries, estimator, file, limit_cols=limit_cols)
        self.model = learner

        # self.tok = Tokenizer(num_words=self.max_words+1) # only num_words-1 will be taken into account!
        # self.model = self.LSTM()
        #
        # X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.15)
        #
        # self.tok.fit_on_texts(X_train)
        # sequences = self.tok.texts_to_sequences(X_train)
        # sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len)
        # y_binary = to_categorical(Y_train)
        # self.model.summary()
        # self.model.fit(sequences_matrix, y_binary, batch_size=self.batch_size, epochs=self.epochs,
        #           validation_split=0.2) #, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]
        #
        # test_sequences = self.tok.texts_to_sequences(X_test)
        # test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=self.max_len)
        #
        # accr = self.model.evaluate(test_sequences_matrix, to_categorical(Y_test))
        # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        self.model.save(self.model_loc)
        #joblib.dump(self.tok, self.tok_loc)

    def LSTM(self):
        model = Sequential()
        model.add(Embedding(self.max_words+1, output_dim=self.max_len))#256))
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
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=12)
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


def transform_text(str, transform_all=True):
    str = str.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    tokens = str.split(r' ')
    new_text = ''
    for token in tokens:
        token = token.lower()
        if re.match(r'^\t', token):  # tab character
            new_text += 'tab '
            token = token.strip('\t')
        if re.match(r'^[0-9][0-9]?$', token):  # one or two digit number
            if transform_all:
                new_text += 'smallNum '
            else:
                new_text += token + ' '
        elif re.match(r'^[0-9][0-9][0-9]$', token):  # three digit number
            new_text += 'mediumNum '
        elif re.match(r'^[0-9]+$', token):  # any digit number (at this point, higher than three digit)
            new_text += 'bigNum '
        elif token.lower() in months:  # name of a month
            new_text += 'month '
        elif re.match(r'^page$', token):  # 'page'
            new_text += 'page '
        elif re.match(r'^appendix$', token):  # 'appendix'
            new_text += 'appendix '
        elif re.match(r'^[a-z]+$', token):  # any letter-only word
            new_text += 'word '
        elif not re.match(r'^(|\s+)$', token):  # any string which is not empty or only whitespace
            new_text += 'mix '
    return new_text


def create_dataset():
    sourcefile = settings.get_dataset_path('marginal_lines')
    texts = pd.read_csv(sourcefile, usecols=['Text'])
    texts = texts.loc[texts['Marginal'] > 0]
    new_text = texts.Text.apply(lambda x: transform_text(x))
    #print(new_text)
    #new_text.to_csv(settings.marginals_id_trans_dataset, index=False)
    return new_text


def run_model():
    nn = NeuralNetwork()
    nn.load_model_from_file()
    df = pd.read_csv(settings.marginals_id_trans_dataset, usecols=['original'])
    #data = df.original
    data = pd.Series(['page 8', 'bhp hello 3', '12 month report', 'epm3424 3 february 1900',
                                 'epm23 february 2000', 'epm34985 4000'])
    p, r = nn.predict(data)#.original)

    for i, row in df.iterrows():
        print(row.original, ', ', p[i], ', ', r[i])


def train(n_queries=10):
    if not os.path.exists(settings.get_dataset_path('page_id')):
        df = create_dataset()
        df.to_csv(settings.get_dataset_path('page_id'))
    nn = NeuralNetwork()
    nn.train(n_queries=n_queries)


def get_page_marginals(marginals):
    if len(marginals) > 0:
        nn = NeuralNetwork()
        p, r = nn.predict(marginals)#.original)
        return r
    else:
        return []


if __name__ == "__main__":
    train()
    #create_dataset()
    #nn = NeuralNetwork()
    #nn.train()
    #run_model()

    # result
    # [[1.93149030e-01 8.52303803e-01]
    #  [1.55359507e-04 9.99890804e-01]
    #  [7.03883052e-01 3.61839056e-01]
    #  [9.63378191e-01 3.04489434e-02]
    #  [8.78076196e-01 1.08638585e-01]
    #  [9.87653494e-01 1.31420493e-02]
    #  [9.74116623e-01 2.65470557e-02]]
    # ------------------
    # [1 1 0 0 0 0 0]