
# import machine_learning_helper as mlh

import pandas as pd
import numpy as np
import settings
import os
import pickle


def data_prep(data, limit_cols=None, y_column=None):  # y=False,
    if y_column in limit_cols:
        limit_cols.remove(y_column)  # don't include y_column in limit cols
    X = data
    if limit_cols:
        for col in limit_cols:
            try:
                X = X.drop(columns=[col])
            except:
                pass
                #print('column ', col, " doesn't exist in X")  # makes it ok to accidentally have multiple of the same col in limit_cols
    if y_column:
        X = X.drop(columns=[y_column])
        Y = data[y_column]
        return X, Y
    return X


def assign_y(x, prev, y_column, line=False, page=True):
    d = int(x['DocID'])
    dunion = (prev['DocID'] == d)
    punion = True
    lunion = True
    if page:
        p = int(x['PageNum'])
        punion = (prev['PageNum'] == p)
    if line:
        l = int(x['LineNum']) - 1
        lunion = (prev['LineNum'] == l)
        #y = prev[y_column].loc[(prev['DocID'] == d) & (prev['PageNum'] == p) & (prev['LineNum'] == l)]

    y = prev[y_column].loc[dunion & punion & lunion]  # need to test

    if len(y) == 0:
        return None
    elif len(y) == 1:
        return y.values[0]
    else:
        print("more rows than 1")  # very possible now that multiple toc pages are possible and legacy doesn't have pagenum to compare against
        print(y.values)


def add_legacy_y(prev_dataset, df, y_column, line=False, page=True):
    if os.path.exists(prev_dataset):
        prev = pd.read_csv(prev_dataset, dtype=int)
        df[y_column] = df.apply(lambda x: assign_y(x, prev, y_column, line, page), axis=1)
        if 'Marginal' in y_column:
            df[y_column].loc[df[y_column] == 2] = 1  # removing the [2] class
        df['TagMethod'].loc[df[y_column] == df[y_column]] = "legacy"
    return df


def classify(data, model_name, limit_cols, mode=settings.dataset_version):
    with open(settings.get_model_path(model_name, mode), "rb") as file:
        model = pickle.load(file)
    data = data_prep(data, limit_cols=limit_cols)
    pred = model.predict(data)
    return pred

#
# def classify(data, model_file, training_function):
#     if not os.path.exists(model_file):
#         training_function()  # need this to act like a passive learner/automatic tagger
#     with open(model_file, "rb") as file:
#         model = pickle.load(file)
#     data = data_prep(data)
#     pred = model.predict(data)
#     return pred