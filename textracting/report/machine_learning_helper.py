## @file
# Module: Major functions for machine learning
# by Anna Andraszek

import pandas as pd
import numpy as np
import paths
import os
import pickle
import inspect
import joblib


## Preps a dataset by removing unecessary columns, separating x and y values
# @param data A pandas DataFrame dataset
# @param limit_cols Column names to be excluded from x
# @param y_column Name of the column containing y. Also determines if Y is returned
# @return X, Y (if y_column)
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

## Finds existing label from previous dataset for same unlabelled sample
# @param x A pandas DataFrame dataset containing x to be labelled
# @param prev A pandas DataFrame dataset from which to take labels
# @param line If line numbers identify members of the dataset
# @param page If page numbers identify members of the dataset
# @param table If table numbers identify members of the dataset
# @return label, or None if previous label isn't found or there are multiple matches
def assign_y(x, prev, y_column, line=False, page=True, table=False):
    d = int(x['DocID'])
    dunion = (prev['DocID'] == d)
    punion = True
    lunion = True
    tunion = True
    if page:
        p = int(x['PageNum'])
        punion = (prev['PageNum'] == p)
    if line:
        l = int(x['LineNum']) - 1
        lunion = (prev['LineNum'] == l)
        #y = prev[y_column].loc[(prev['DocID'] == d) & (prev['PageNum'] == p) & (prev['LineNum'] == l)]
    if table:
        t = int(x['TableNum'])
        tunion = (prev['TableNum'] == t)
    y = prev[y_column].loc[dunion & punion & lunion & tunion]  # need to test

    if len(y) == 0:
        return None
    elif len(y) == 1:
        return y.values[0]
    else:
        print("more rows than 1")  # very possible now that multiple toc pages are possible and legacy doesn't have pagenum to compare against
        print(y.values)


## When working with a new or unlabelled version of a dataset, migrates labels from previous version
# @param prev_dataset Filename of the previous dataset
# @param df Current dataset as a pandas DataFrame
# @param y_column Name of the column containing y
# @param line If line numbers identify members of the dataset
# @param page If page numbers identify members of the dataset
# @param table If table numbers identify members of the dataset
# @return labelled DataFrame
def add_legacy_y(prev_dataset, df, y_column, line=False, page=True, table=False):
    if os.path.exists(prev_dataset):
        prev = pd.read_csv(prev_dataset)  #, dtype=int)
        df[y_column] = df.apply(lambda x: assign_y(x, prev, y_column, line, page, table), axis=1)
        if 'Marginal' in y_column:
            df[y_column].loc[df[y_column] == 2] = 1  # removing the [2] class
        df['TagMethod'].loc[df[y_column] == df[y_column]] = "legacy"
    return df


## Predicts labels for x and returns, with probabilities
# @param data A pandas DataFrame dataset for which to predict
# @param model_name Identifying name of the model to use
# @param y_column Name of column containing y
# @param limit_cols Column names to exclude from x
# @param mode A string indicating the 'version' of the model - can be a name, or 'production'
# @return pred, proba: Arrays of predictions and probabilities
def classify(data, model_name, y_column, limit_cols, mode=paths.dataset_version):
    #frame = inspect.stack()[9]  # 9 only works if this functions is called from get_classified  # 1 if called from model file
    model_path = paths.get_model_path(model_name, mode)
    if not os.path.exists(model_path):
        frame = inspect.stack()[2]  # 0: this, 1: mlh.get_classified, 2: model file
        module = inspect.getmodule(frame[0])  # inspect.getmodule(frame[0])  # gets the module that this function was called from to call the correct training function
        module.train(n_queries=0, mode=mode, spec_name=model_name) #datafile=settings.get_dataset_path(model_name, mode), model_file=model_path)
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    if isinstance(data, pd.DataFrame) and y_column in data.columns:
        limit_cols.append(y_column)  #better than passing y_column to data prep to be removed because then y will also be returned
    data = data_prep(data, limit_cols)
    pred = model.predict(data)
    proba = model.predict_proba(data)
    #print(proba)
    return pred, proba

## Predicts labels for x and returns, with probabilities
# @param data A pandas DataFrame dataset for which to predict
# @param model_name Identifying name of the model to use
# @param y_column Name of column containing y
# @param limit_cols Column names to exclude from x
# @param mode A string indicating the 'version' of the model - can be a name, or 'production'
# @param masked Boolean, if the result should be returned with a mask of y=1
# @return data, and classes of the data if masked=False
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