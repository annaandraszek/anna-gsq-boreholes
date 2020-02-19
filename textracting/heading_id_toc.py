## @file
# Module functions for finding headings in the table of contents

from toc_classification import get_toc_pages
import glob
import json
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import tensorflow as tf
import joblib
import os
from keras.models import Sequential
import numpy as np
import pandas as pd
import re
import settings
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import label_binarize
import active_learning
import machine_learning_helper as mlh

name = 'heading_id_toc'
y_column = 'Heading'
columns = ['DocID', 'PageNum', 'LineNum', 'Text', 'SectionPrefix', 'SectionText', 'SectionPage', 'ProcessedText',
           'Heading', 'TagMethod']
initial_columns = ['DocID', 'PageNum', 'LineNum', 'Text', 'Heading', 'TagMethod']
limit_cols = ['DocID', 'PageNum', 'LineNum', 'Text', 'SectionPrefix', 'SectionText', 'SectionPage', 'TagMethod']


class Text2Seq(TransformerMixin, BaseEstimator):
    def __init__(self, classes=3, pad_len=None):
        self.tok = Tokenizer()
        #self.labelbin = LabelBinarizer()
        self.classes = range(classes)
        self.pad_len = pad_len

    def fit(self, x, y=None):
        if isinstance(x, list):  # when the AL does predict proba, gives a single sample inside a 1-element list
            if len(x) != 1:
                print("x is longer than one sample, ", x)  # check in case it gives multiple samples and this code is wrong??
            x = x[0]
        if isinstance(x, np.ndarray):
            x = pd.Series(data=x.T[0])
        if isinstance(x, pd.DataFrame):  # may be df or ndarray
            try:
                x = x['ProcessedText']
            except KeyError:  # accounts for how it was in the old version
                x = x['SectionText']
        self.tok.fit_on_texts(x)  # have to specify the column to give it a series

        #if y is not None:
        #    self.labelbin.fit(y)

        return self

    def transform(self, x, y=None):
        if isinstance(x, pd.DataFrame):  # may be df or ndarray
            try:
                x = x['ProcessedText']
            except KeyError:
                x = x['SectionText']
        if isinstance(x, list):  # when the AL does predict proba, gives a single sample inside a 1-element list
            if len(x) != 1:
                print("x is longer than one sample, ", x)  # check in case it gives multiple samples and this code is wrong??
            x = x[0]
        if isinstance(x, np.ndarray):
            x = pd.Series(data=x.T[0])  # make array 1D before it can be a series
        sequences = self.tok.texts_to_sequences(x)
        if self.pad_len is not None:
            sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.pad_len)
        else:
            sequences_matrix = sequence.pad_sequences(sequences)

        if y:
            y_binary = label_binarize(y, self.classes)
            #y_binary = self.labelbin.transform(y)
            return sequences_matrix, y_binary

        return sequences_matrix


class NeuralNetwork(): #give this arguments like: model type, train/test file
    #max_words = 900
    #max_len = 15
    #y_dict = {}
    epochs = 20
    batch_size = 30
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    #model_path = settings.model_path
    #model_name = 'heading_id_cyfra1'

    def __init__(self, mode=settings.dataset_version):
        #self.model_name = 'heading_id_' + model_name
        self.model_loc = settings.get_model_path(name, mode) #self.model_path + self.model_name + '.h5'
        self.mode = mode
        #self.tok_loc = settings.get_model_path('heading_id_toc', tokeniser=True)#self.model_path + self.model_name + 'tokeniser.joblib'
        print()

    def train(self, n_queries=10):  #settings.dataset_path + 'processed_heading_id_dataset_cyfra1.csv'):
        datafile = settings.get_dataset_path(name, self.mode)
        df = pd.read_csv(datafile)
        self.max_words, self.max_len = check_maxlens(df)

        lstm = KerasClassifier(build_fn=self.LSTM, batch_size=self.batch_size, epochs=self.epochs,
                  validation_split=0.2)

        clf = Pipeline([
            ('transform', Text2Seq()),
            ('lstm', lstm)
        ], verbose=True)

        y_column = 'Heading'
        if self.mode == settings.production:
            global limit_cols  # to be able to use this arg in .train
            limit_cols = ['DocID', 'LineNum']  # just accounts for old version
        accuracy, learner = active_learning.train(df, y_column, n_queries, clf, datafile, limit_cols=limit_cols)
        self.model = learner
        with open(self.model_loc, "wb") as file:
            joblib.dump(learner, file)
        # joblib.dump(self.tok, self.tok_loc)

        print("End of training stage. Re-run to train again")
        return accuracy

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


    def load_model_from_file(self, model_loc=None):
        if model_loc is None:
            model_loc = self.model_loc
        self.model = joblib.load(model_loc)  #load_model(self.model_loc)


    def predict(self, strings, encode=False, mode=settings.dataset_version):
        if encode:
            strings = [num2cyfra1(s) for s in strings]

        predictions = mlh.get_classified(strings, name, y_column, limit_cols, mode)
        return predictions #, np.argmax(predictions, axis=0)


def train(n_queries=10, mode=settings.dataset_version):
    if not os.path.exists(settings.get_dataset_path('proc_heading_id_toc', mode)):
        if not os.path.exists(settings.get_dataset_path(name, mode)):
            create_dataset(datafile=settings.get_dataset_path(name, mode))
        df = pre_process_id_dataset(datafile=settings.get_dataset_path(name, mode))
        df.to_csv(settings.get_dataset_path('proc_heading_id_toc'), mode)

    nn = NeuralNetwork(mode)
    nn.train(n_queries=n_queries)


def num2cyfra1(string):
    s = ''
    prev_c = ''
    i = 1
    for c in string:
        if re.match(r'[0-9]', c):
            if prev_c != 'num':
                if c == '0':
                    if prev_c == '.':
                        continue
                    else:
                        s += 'cyfra' + str(i) + ' '
                        i += 1
                        prev_c = 'num'
                else:
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


def pre_process_id_dataset(datafile=settings.get_dataset_path('heading_id_toc'), training=True):
    if isinstance(datafile, pd.DataFrame):
        df = datafile
        #df['LineText'] = df['Text']
    else:
        df = pd.read_csv(datafile)
    # break up the LineText column into SectionPrefix, SectionText, and SectionPage
    #newdf = pd.DataFrame(columns=['DocID', 'PageNum', 'LineNum', 'SectionPrefix', 'SectionText', 'SectionPage'])
    #newdf.DocID = df.DocID
    #if 'PageNum' in df.columns.values:
    #    newdf.PageNum = df.PageNum
    #newdf.LineNum = df.LineNum
    #if training:
    #    newdf['Heading'] = df.Heading
    #    newdf['TagMethod'] = df.TagMethod

    df['SectionPrefix'], df['SectionText'] = zip(*df.Text.map(split_prefix))
    df['SectionText'], df['SectionPage'] = zip(*df.SectionText.map(split_pagenum))

    df.SectionPrefix = df.SectionPrefix.apply(lambda x: num2cyfra1(x))
    df.SectionPage = df.SectionPage.apply(lambda x: num2cyfra1(x))

    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True, subset=['SectionText'])
    df.replace(np.nan, '', inplace=True)  # nan values cause issues when adding columns

    df['ProcessedText'] = df.SectionPrefix + df.SectionText + df.SectionPage
    #newdf.drop(axis=1, columns=['SectionPrefix', 'SectionPage'], inplace=True)
    return df


def create_dataset(datafile=settings.get_dataset_path(name)):
    df = pd.DataFrame(columns=columns)
    lines_docs = sorted(glob.glob('training/restructpageinfo/*'))
    toc_df = pd.read_csv(settings.get_dataset_path('toc'))
    toc_pages = get_toc_pages(toc_df)
    for lines_doc in lines_docs:
        pages = json.load(open(lines_doc))
        docid = int(lines_doc.split('\\')[-1].replace('_1_restructpageinfo.json', '').strip('cr_'))
        tocpg = toc_pages.loc[toc_pages['DocID'] == docid]
        try:
            page = tocpg.PageNum.values[0]
            for lines in pages.items():
                if lines[0] == str(page):
                    docset = []
                    for line, i in zip(lines[1], range(len(lines[1]))):
                        heading = 0
                        if re.match(r'^([0-9]+\.[0-9]+\s+\w+)', line['Text']):
                            heading = 2
                        elif re.match(r'^[0-9]+\.*\s+\w+', line['Text']):
                            heading = 1

                        docset.append([docid, page, i+1, line['Text'], heading, None])
                    pgdf = pd.DataFrame(data=docset, columns=initial_columns)
                    df = df.append(pgdf, ignore_index=True)
        except IndexError:
            print("IndexError ", tocpg, docid)
    prev_dataset = settings.get_dataset_path(name, settings.production)
    df = mlh.add_legacy_y(prev_dataset, df, y_column, line=True, page=False)  # page not present in legacy dataset
    df = pre_process_id_dataset(df)
    df.to_csv(datafile, index=False)
    return df


def get_toc_headings(df, mode=settings.dataset_version, masked=False):
    return mlh.get_classified(df, name, y_column, limit_cols, mode, masked)


if __name__ == '__main__':
    create_dataset()
    #pre = 'cyfra1strona'
    # df = pre_process_id_dataset()  #pre)
    # df.to_csv(settings.get_dataset_path('proc_heading_id_toc'), index=False)
    # #df.to_csv(settings.dataset_path + 'processed_heading_id_dataset_' + pre + '.csv', index=False)
    #
    # data = settings.get_dataset_path('proc_heading_id_toc')
    nn = NeuralNetwork()
    nn.train()
    # # nn.load_model_from_file()
    # p, r = nn.predict(['4.0 drilling', 'Introduction 1', 'lirowjls', 'figure drilling', '5 . 9 . geology of culture 5', '1 . 0 introduction', '8 . 1 introduction 7'], encode=True)
    # #     #['4.3 drilling', 'Introduction strona', 'lirowjls', 'figure drilling', '5 . 9 . geology of culture strona', '1 . introduction', '8 . 1 introduction strona'])
    # print(p)
    # print('------------------')
    # print(r)

    #create_identification_dataset()
    # df = pre_process_id_dataset()
    # df.to_csv(settings.get_dataset_path('proc_heading_id_toc'), index=False)