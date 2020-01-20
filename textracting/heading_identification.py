from toc_classification import get_toc_pages
import glob
import json
from keras.layers import LSTM, Activation, Dense, Dropout, TimeDistributed, Input, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import load_model
import tensorflow as tf
import joblib
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import re
import settings


class NeuralNetwork(): #give this arguments like: model type, train/test file
    #max_words = 900
    #max_len = 15
    #y_dict = {}
    epochs = 10
    batch_size = 50
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model_path = settings.model_path
    #model_name = 'heading_id_cyfra1'

    def __init__(self, model_name='cyfra1'):
        self.model_name = 'heading_id_' + model_name
        self.model_loc = self.model_path + self.model_name + '.h5'
        self.tok_loc = self.model_path + self.model_name + 'tokeniser.joblib'

    def train(self, file='processed_heading_id_dataset_cyfra1.csv'):
        df = pd.read_csv(file)
        self.X = df['SectionText']
        self.Y = df['Heading']
        self.max_words, self.max_len = check_maxlens(df)
        self.tok = Tokenizer(num_words=self.max_words)
        self.model = self.LSTM() #StackedLSTM()

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


    def StackedLSTM(self):
        # Stacked LSTM for sequence classification from https://keras.io/getting-started/sequential-model-guide/
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=self.max_len))
        model.add(LSTM(32, return_sequences=True))
        model.add(LSTM(32))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'categorical_accuracy'])
        return model

    def LSTM(self):
        model = Sequential()
        model.add(Embedding(self.max_words, output_dim=256))#self.max_len))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
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

        #encoded = [num2cyfra1(s) for s in strings]
        sequences = self.tok.texts_to_sequences(strings)
        #self.max_len
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=256)
        predictions = self.model.predict(sequences_matrix)
        return predictions, np.argmax(predictions, axis=1)


def num2cyfra1(string):
    s = ''
    prev_c = ''
    i = 1
    for c in string:
        if re.match(r'[0-9]', c):
            if prev_c != 'num':
                s += 'cyfra' + str(i) + ' '
                i += 1
                prev_c = 'num'
        elif c == '.':
            s += 'punkt '
            prev_c = '.'
    return s


def check_maxlens(df):
    series = df['SectionText']
    max_len = series.str.len().max()
    words = series.str.lower().str.split().apply(set().update)
    max_words = len(words)
    return max_words, max_len


def split_prefix(string):
    s = re.split(r'(^[0-9]+\.*[0-9]*\.*[0-9]*)', string, 1)
    if len(s) == 1:
        s = ['', s[0]]
    elif len(s) == 3:
        s = [s[-2], s[-1]]
    return s


def split_pagenum(string):
    s = re.split(r'(\t[0-9]+$)', string, 1) # if $ doesn't work try \Z
    if len(s) == 1:
        s = [s[0], '']
    elif len(s) == 3:
        s = [s[0], s[1]]
    return s


def num2cyfra(string):
    s = ''
    prev_c = ''
    for c in string:
        if re.match(r'[0-9]', c):
            if prev_c != 'num':
                s += 'cyfra '
                prev_c = 'num'
        elif c == '.':
            s += 'punkt '
            prev_c = '.'
    return s


def num2strona(string):
    s = ''
    prev_c = ''
    for c in string:
        if re.match(r'[0-9]', c):
            if prev_c != 'num':
                s += 'strona '
                prev_c = 'num'
    return s


def pre_process_id_dataset(pre='cyfra', datafile=settings.dataset_path + "heading_id_dataset.csv", training=True):
    if isinstance(datafile, pd.DataFrame):
        df = datafile
        df['LineText'] = df['Text']
    else:
        df = pd.read_csv(datafile)
    # break up the LineText column into SectionPrefix, SectionText, and SectionPage
    newdf = pd.DataFrame(columns=['DocID', 'LineNum', 'SectionPrefix', 'SectionText', 'SectionPage'])
    newdf.DocID = df.DocID
    newdf.LineNum = df.LineNum
    if training:
        newdf['Heading'] = df.Heading

    newdf.SectionPrefix, newdf.SectionText = zip(*df.LineText.map(split_prefix))
    newdf.SectionText, newdf.SectionPage = zip(*newdf.SectionText.map(split_pagenum))

    if 'cyfra1' in pre:
        newdf.SectionPrefix = newdf.SectionPrefix.apply(lambda x: num2cyfra1(x))
        newdf.SectionPage = newdf.SectionPage.apply(lambda x: num2cyfra1(x))
    else:
        newdf.SectionPrefix = newdf.SectionPrefix.apply(lambda x: num2cyfra(x))
        newdf.SectionPage = newdf.SectionPage.apply(lambda x: num2cyfra(x))

    if 'strona' in pre:
        newdf.SectionPage = newdf.SectionPage.apply(lambda x: num2strona(x))

    newdf.replace('', np.nan, inplace=True)
    newdf.dropna(inplace=True, subset=['SectionText'])
    newdf.replace(np.nan, '', inplace=True)  # nan values cause issues when adding columns

    newdf.SectionText = newdf.SectionPrefix + newdf.SectionText + newdf.SectionPage
    newdf.drop(axis=1, columns=['SectionPrefix', 'SectionPage'], inplace=True)
    return newdf


def create_identification_dataset():
    df = pd.DataFrame(columns=['DocID', 'LineNum', 'LineText', 'Heading'])
    lines_docs = sorted(glob.glob('training/restructpagelines/*'))
    toc_pages = get_toc_pages()
    for lines_doc in lines_docs:
        pages = json.load(open(lines_doc))
        docid = int(lines_doc.split('\\')[-1].replace('_1_restructpagelines.json', '').strip('cr_'))
        tocpg = toc_pages.loc[toc_pages['DocID'] == docid]
        try:
            page = tocpg.PageNum.values[0]
            for lines in pages.items():
                if lines[0] == str(page):
                    docset = []
                    for line, i in zip(lines[1], range(len(lines[1]))):
                        heading = 0
                        if re.match(r'^([0-9]+\.[0-9]+\s+\w+)', line):
                            heading = 2
                        elif re.match(r'^[0-9]+\.*\s+\w+', line):
                            heading = 1

                        docset.append([docid, i, line, heading])
                    pgdf = pd.DataFrame(data=docset, columns=['DocID', 'LineNum', 'LineText', 'Heading'])
                    df = df.append(pgdf, ignore_index=True)
        except IndexError:
            print("IndexError ", tocpg, docid)
    df.to_csv("heading_id_dataset.csv", index=False)
    return df


if __name__ == '__main__':
    #df = create_identification_dataset()
    #pre = 'cyfra1strona'
    #df = pre_process_id_dataset(pre)
    #df.to_csv('processed_heading_id_dataset_' + pre + '.csv', index=False)

    data = 'processed_heading_id_dataset_cyfra1.csv'
    nn = NeuralNetwork()
    #nn.train(data)
    nn.load_model_from_file()
    p, r = nn.predict(['4.3 drilling', 'Introduction 1', 'lirowjls', 'figure drilling', '5 . 9 . geology of culture 5', '1 . introduction', '8 . 1 introduction 7'])
        #['4.3 drilling', 'Introduction strona', 'lirowjls', 'figure drilling', '5 . 9 . geology of culture strona', '1 . introduction', '8 . 1 introduction strona'])
    print(p)
    print('------------------')
    print(r)