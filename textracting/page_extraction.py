from __future__ import print_function
import pandas as pd
import settings
from page_identification import transform_text
import page_identification
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import load_model
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Flatten
from keras.utils import to_categorical
import numpy as np
os.environ['KMP_WARNINGS'] = '0'
from keras.models import Sequential
from keras import layers
from six.moves import range
import re


def num2word(str):
    new_str = ''
    for token in str.split():
        if re.match(r'[0-9]+', token):
            new_str += 'smallNum '
        else:
            new_str += token + ' '
    return new_str


def check_maxlens(x):
    seqs = x.str.lower().str.split()
    max_seq_len = len(max(seqs, key=lambda x:len(x)))
    all_words = []
    seqs.apply(lambda x: all_words.extend(x))
    unique_words = len(set(all_words))
    #max_words = len(words.unique())
    return unique_words, max_seq_len


class NeuralNetwork():
    epochs = 100
    batch_size = 30
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model_path = settings.model_path

    def __init__(self, model_name='mask_nn', model_type='NN'):
        self.model_name = 'page_ex_' + model_name
        self.model_loc = self.model_path + self.model_name + '.h5'
        self.tok_loc = self.model_path + self.model_name + 'tokeniser.joblib'
        self.classes_loc = self.model_path + self.model_name + 'class_dict.joblib'
        self.mode_type = model_type

    def train(self, file=settings.page_extraction_dataset):
        df = pd.read_csv(file)
        self.X = df['transformed']
        self.Y = df['position']         # try with y position instead of y value
        self.X = self.X.apply(lambda x: num2word(x))
        self.max_words, self.max_len = check_maxlens(self.X)
        self.classes, y_vectorised = self.position2int()
        self.inv_classes = {v: k for k, v in self.classes.items()}
        y_masked = np.zeros((self.Y.size, self.max_len))
        for i, j in zip(y_masked, y_vectorised):
            p = self.inv_classes[j]
            i[p] = 1

        self.num_classes = len(self.classes.items())
        self.tok = Tokenizer(num_words=self.max_words+1) # only num_words-1 will be taken into account!

        if self.mode_type == 'LSTM':
            self.model = self.LSTM()
        else:
            self.model = self.NN()

        X_train, X_test, Y_train, Y_test = train_test_split(self.X, y_masked, test_size=0.15)

        self.tok.fit_on_texts(self.X)
        sequences = self.tok.texts_to_sequences(X_train)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len)
        #y_binary = to_categorical(Y_train) # y needs to be onehot encoded

        self.model.summary()
        self.model.fit(sequences_matrix, Y_train, batch_size=self.batch_size, epochs=self.epochs,
                  validation_split=0.2) #, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]

        test_sequences = self.tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=self.max_len)

        accr = self.model.evaluate(test_sequences_matrix, Y_test)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        self.model.save(self.model_loc)
        joblib.dump(self.tok, self.tok_loc)
        joblib.dump(self.inv_classes, self.classes_loc)

    def NN(self):
        model = Sequential()
        model.add(Embedding(input_length=self.max_len, input_dim = self.max_words+1, output_dim=self.max_len))#256))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        #model.add(Dense(256, activation='relu', input_shape=()))
        model.add(Flatten())
        model.add(Dense(self.max_len, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    def LSTM(self):
        model = Sequential()
        model.add(Embedding(input_length=self.max_len, input_dim=self.max_words + 1, output_dim=self.max_len))  # 256))
        #model.add(LSTM(48, return_sequences=True))
        model.add(LSTM(192))
        model.add(Dropout(0.1))
        #model.add(Flatten())
        model.add(Dense(self.max_len, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model



    def load_model_from_file(self):
        self.model = load_model(self.model_loc)
        self.tok = joblib.load(self.tok_loc)
        self.inv_classes = joblib.load(self.classes_loc)
        self.model._make_predict_function()


    def predict(self, strings):
        if not os.path.exists(self.model_loc):
            self.train()
        try:
            self.model
        except AttributeError:
            self.load_model_from_file()

        original_strings = strings
        strings = strings.apply(lambda x: num2word(x))

        sequences = self.tok.texts_to_sequences(strings)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=12)
        predictions = self.model.predict(sequences_matrix)
        reverse_predictions = np.flip(predictions, 1)
        pred_classes = np.argmax(reverse_predictions, axis=1)
        classes = np.array([-(x+1) for x in pred_classes])
        #classes = [self.inv_classes[x] for x in pred_classes]
        texts = original_strings.apply(lambda x: x.split())
        #for i, j in zip(texts, classes):
        #    print(i[j])
        pagenums = [x[y] for x, y in zip(texts, classes)]
        return pagenums #, predictions

    def position2int(self, create_map=False):
        y = self.Y
        map = {}
        vector = []
        i = 0
        for e in y:
            if e not in map.keys():
                map[e] = i
                i += 1
            vector.append(map[e])
        return map, vector


# goal of this is to match page numbers in the text with pages of the pdf, by which we navigate - so we can navigate to
#   the textually-numbered pages
# therefore essential to return a mapping between the actual page number and the textual page number - which needs to
#   be done a few steps ahead of this to carry it through to here; in marginals identification
def get_page_nums(marginals, ml_only=False): # given a dataset of marginals of the document
    page_marginals_mask = page_identification.get_page_marginals(marginals.Text)  # from all the marginals, get only those containing page numbers
    page_marginals = marginals.loc[page_marginals_mask == 1]
    nn = NeuralNetwork('mask_nn', 'NN')
    trans_marginals = page_marginals.Text.apply(lambda x: transform_text(x, transform_all=False))
    if ml_only:
        page_nums = nn.predict(trans_marginals)
    else:  # if there is only one smallNum in the line, return that. if there are more, put it through the predict function
        page_nums = []
        for line in trans_marginals:
            tokens = line.split()
            page_num = None
            for token in tokens:
                if re.match('^[0-9]+$', token):
                    if page_num:  # have found two nums in the line, need to go to predict
                        page_num = nn.predict(pd.Series(line))[0]
                        break
                    else:
                        page_num = token
            page_nums.append(page_num)
    page_marginals['Page'] = page_nums
    return page_marginals


def run_model(model_name, model_type='NN'):
    nn = NeuralNetwork(model_name, model_type)
    data = pd.Series(['page 3 of 8',
                      'bhp hello 3',
                      'epm3424 \t3 \tfebruary 1900',
                      'epm3424 \tpage 3 \tfebruary 1900',
                      'epm3424 page \t3 \tfebruary 1900',
                      'epm3424 page 3 \tfebruary 1900',
                      'epm34985 \t40',
                      '8 \t9 \t10',
                      '8 may 1998 \treport 90',
                      '3 \tbhp annual report'])
    trans_data = data.apply(lambda x: transform_text(x, transform_all=False))
    r = nn.predict(trans_data)
    print(r)

    all_predictions = False
    if all_predictions:
        df = pd.read_csv(settings.page_extraction_dataset)
        data = df.transformed
        r = nn.predict(data)
        correct = 0
        incorrect = 0
        for i, row in df.iterrows():

            print(row.original, ', ', r[i])
            if str(row.pagenum) != r[i]:
                incorrect += 1
            else:
                correct += 1
        print('real accuracy: ', correct/(correct+incorrect))


def create_dataset():
    sourcefile = settings.marginals_id_trans_dataset
    texts = pd.read_csv(sourcefile)
    page_texts = texts.loc[texts.tag == 1]
    page_texts.transformed = page_texts.original.apply(lambda x: transform_text(x, transform_all=False))

    page_texts = page_texts.drop(['tag'], axis=1)
    page_texts['pagenum'] = 0

    page_texts.to_csv(settings.page_extraction_dataset, index=False)


if __name__ == '__main__':
    #create_dataset()
    #nn = NeuralNetwork(model_name='mask_lstm', model_type='LSTM')
    #nn.train()
    run_model(model_name='mask_lstm', model_type='LSTM')

    #nn = NeuralNetwork(model_name='mask_nn', model_type='NN')
    #nn.train()
    run_model(model_name='mask_nn', model_type='NN')