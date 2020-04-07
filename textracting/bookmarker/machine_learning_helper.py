
# import machine_learning_helper as mlh

import pandas as pd
import numpy as np
import settings
import os
import pickle
import inspect
import joblib


def data_prep(data, limit_cols=None, y_column=None):  # y=False,
    if y_column in limit_cols:
        limit_cols.remove(y_column)  # don't include y_column in limit cols
    if 'TagMethod' not in limit_cols:
        limit_cols.append('TagMethod')
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
        prev = pd.read_csv(prev_dataset)  #, dtype=int)
        df[y_column] = df.apply(lambda x: assign_y(x, prev, y_column, line, page), axis=1)
        if 'Marginal' in y_column:
            df[y_column].loc[df[y_column] == 2] = 1  # removing the [2] class
        df['TagMethod'].loc[df[y_column] == df[y_column]] = "legacy"
    return df


def classify(data, model_name, y_column, limit_cols, mode=settings.dataset_version):
    #frame = inspect.stack()[9]  # 9 only works if this functions is called from get_classified  # 1 if called from model file
    model_path = settings.get_model_path(model_name, mode)
    if not os.path.exists(model_path):
        frame = inspect.stack()[2]  # 0: this, 1: mlh.get_classified, 2: model file
        module = inspect.getmodule(frame[0])  # inspect.getmodule(frame[0])  # gets the module that this function was called from to call the correct training function
        module.train(n_queries=0, mode=mode)#datafile=settings.get_dataset_path(model_name, mode), model_file=model_path)
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    if isinstance(data, pd.DataFrame) and y_column in data.columns:
        limit_cols.append(y_column)  #better than passing y_column to data prep to be removed because then y will also be returned
    data = data_prep(data, limit_cols)
    pred = model.predict(data)
    proba = model.predict_proba(data)
    #print(proba)
    return pred, proba


def get_classified(data, model_name, y_column, limit_cols, mode, masked=True):
    classes, proba = classify(data, model_name, y_column, limit_cols, mode)  #classify_page(df)
    if masked:
        mask = np.array([True if i>0 else False for i in classes])
    if isinstance(data, pd.DataFrame):
        data[y_column] = classes
        data['proba'] = proba[:, 1]
        if masked:
            data = data[mask]
        return data
    else:
        if masked:
            data = data[mask]
            classes = classes[mask]
        return data, classes
#
# def classify(data, model_file, training_function):
#     if not os.path.exists(model_file):
#         training_function()  # need this to act like a passive learner/automatic tagger
#     with open(model_file, "rb") as file:
#         model = pickle.load(file)
#     data = data_prep(data)
#     pred = model.predict(data)
#     return pred