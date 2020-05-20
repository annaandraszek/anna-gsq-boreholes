## @file
# Active learning module: Functions for facilitating active (and passive) learning
# by Anna Andraszek

import time
from IPython import display
import paths
from PIL import Image, ImageDraw

import textractor.textloading
from textractor import textloading
import re
import img2pdf
from pdf2image import convert_from_path, exceptions
import os
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json
import sklearn
from report import machine_learning_helper as mlh
from borehole import tables
import random


## Gets input from the console and checks its validity
# @param classes array of ints representing classes
# @return valid user input for a class
def get_input(classes):
    y = -1
    while y not in classes:
        print("Enter one of: ", str(classes))
        y = input()
        try:
            y = int(y)  # set it as int here instead of on input to avoid error breaking execution when input is bad
        except:
            continue
    return y


## Displays part of a page in python notebook
# @param docid Unique identifying int of report
# @param page page number
# @param line line number
# @param mode A string indicating the 'version' of the model - can be a name, or 'production'
def display_page(docid, page, line=None, mode=paths.dataset_version):
    pg_path = paths.get_report_page_path(int(docid), int(page))  # docid, page
    image = Image.open(pg_path)
    width, height = image.size

    if line:
        draw = ImageDraw.Draw(image, 'RGBA')
        draw.line([(1, 1), (1, height-3)], fill="blue", width=3)  # draw parallel lines down the page
        draw.line([(width-3, 1), (width-3, height-3)], fill="blue", width=3)

        docinfofile = paths.get_restructpageinfo_file(docid)
        docinfo = json.load(open(docinfofile, "r"))
        pageinfo = docinfo[str(page)]
        lineinfo = pageinfo[int(line)-1]  #-1 because linenum starts from 1

        box = lineinfo['BoundingBox']
        ln_left = width * box['Left']
        ln_top = height * box['Top']

        crop_height = height / 3
        left = 0
        right = width
        top = ln_top - box['Height'] - (crop_height/2)  # bottom > top  bc of coordinate system
        bottom = ln_top + (crop_height/2)

        if top < 0:  # if top is outside of bounds, add to it to make it 0, and sub from bottom
            change = top
            top = 0
            bottom -= change
            draw.line([(1, 1), (width-3, 1)], fill="blue", width=3)

        elif bottom > height:
            change = bottom - height
            bottom = height
            top -= change
            draw.line([(1, height-3), (width-3, height-3)], fill="blue", width=3)

        draw.rectangle([ln_left, ln_top, ln_left + (width * box['Width']), ln_top + (height * box['Height'])], outline='green', width=2)

        crop_image = image.crop((left, top, right, bottom))
        #crop_ln_top = crop_height * box['Top']

        #draw.rectangle([ln_left, crop_ln_top, ln_left + (width * box['Width']), crop_ln_top + (crop_height * box['Height'])], outline='green')
        image = crop_image

    display.display(image)
    # line option: draw a box around the line
    # get docinfo, query the line number and bounding box
    # crop page to about 1/3 of it to make it more focused on the line

    print(pg_path)
    if line: print("line: ", line)

## Used instead of uncertainty sampling when classifying borehole tables, because uncertainty sampling will sample inputs with UNK tokens.
# This sampling method looks for specific words and gets a random set of n_queries of inputs containing these.
# @param pool Dataframe of unlabelled samples
# @param n_queries how many are to be sampled
# @return idx, inst: indices of samples and samples
def borehole_sample(pool, n_queries):
    hole_words = ['hole', 'bore', 'well', 'core']#, 'drill']
    hole_pool = []
    for word in hole_words:
        word_pool = [(x, idx) for x, idx in zip(pool.iloc[:,0], pool.index.values) if word in x]
        hole_pool.extend(word_pool)
    hole_pool = set(hole_pool)
    hole_pool = list(hole_pool)
    sample = random.sample(hole_pool, n_queries)
    idx = np.asanyarray([i[1] for i in sample])
    inst = np.asanyarray([i[0] for i in sample])
    #idx.shape = (idx.shape[0], 1)
    inst.shape = (inst.shape[0], 1)
    return idx, inst

## Allows user to annotate data in notebook, when there is unlabelled data, and then train the model with those annotations included.
# Data given to label is that which has most uncertain label to the model.
# Can be used to try to incrementally improve model with previously unlabelled data, but can also be a good way of labelling
# data as it presents it in its original context, wheras the contents of the csv the dataset resides in may be also transformed / not have context.
# Can be slow - as it gets and displays images.
# Must be run in python notebook to view images. (However, can change display function to disply image in pop up.)
# @param data A dataset stored as a pandas DataFrame
# @param n_queries Number of points of data to label in active learning
# @param y_column The name of the column containing y values
# @param estimator The model to train. Currently use scikit-learn and keras models.
# @param limit_cols Columns in the dataset to exclude from x
# @param mode A string indicating the 'version' of the model - can be a name, or 'production'
# @return labelled data, accuracy of model, trained model
def active_learning(data, n_queries, y_column, estimator=RandomForestClassifier(), limit_cols=None, mode=paths.dataset_version):
    line = False
    if y_column in ['Marginal', 'Heading']:  # covers marginal_lines, heading_id_toc, heading_id_intext
        line = True  # determines if a line or page is to to be displayed
    classes = pd.unique(data[y_column].values)  #todo: check type
    classes = sorted(filter(lambda v: v==v, classes))
    X_initial, Y_initial, X_pool, y_pool, refs = al_data_prep(data, y_column, limit_cols, mode)
    if mode == paths.production:
        test_percentage = 0
    else:
        test_percentage = 0.2
    if 'lstm' in estimator.named_steps:
        test_size = int(X_initial.shape[0] * test_percentage)
        X_train, y_train = X_initial[:-test_size], Y_initial[:-test_size]
        X_test, y_test = X_initial[-test_size:], Y_initial[-test_size:]
    else:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_initial, Y_initial,
                                                                               test_size=test_percentage)
    learner = ActiveLearner(estimator=estimator, #ensemble.RandomForestClassifier(),
                            query_strategy=uncertainty_sampling,
                            X_training=X_train.values, y_training=y_train.astype(int))
    accuracy_scores = [learner.score(X_test, y_test.astype(int))]
    if 'boreholes' not in mode:
        query_idx, query_inst = learner.query(X_pool, n_instances=n_queries)
        query_idx = np.asarray([refs['idx'][i] for i in query_idx])
    else:
        query_idx, query_inst = borehole_sample(X_pool, n_queries)
    y_new = np.zeros(n_queries, dtype=int)
    time.sleep(5)
    for i in range(n_queries):
        idx = query_idx[i]
        #page=int(query_inst[i][0])
        if 'boreholes' not in mode:
            page = refs['pagenums'].loc[idx]
        if line:
            line=refs['linenums'].loc[idx]
        if 'boreholes' in mode:
            page = refs['Tables'].loc[idx]
        y = al_input_loop(learner, query_inst[i], refs['docids'].loc[idx], n_queries, classes, page=page, line=line, mode=mode)
        y_new[i] = y
        #print("index: ", idx)
        #print("x: ", data.at[idx, 'Columns'])
        data.at[idx, y_column] = y  # save value to copy of data
        data.at[idx, 'TagMethod'] = 'manual'

    learner.teach(query_inst, y_new)  # reshape 1, -1
    accuracy_scores.append(learner.score(X_test, y_test.astype(int)))
    preds = learner.predict(X_test)
    #print("End of annotation. Samples, predictions, annotations: ")
    #print(ref_docids.iloc[query_idx].values,
    #      np.concatenate((query_inst, np.array([preds]).T, y_new.reshape(-1, 1)), axis=1))
    print(sklearn.metrics.confusion_matrix(preds, y_test.astype(int)))
    accuracy = accuracy_scores[-1]
    print(accuracy)
    return data, accuracy, learner

## When there is no unlabelled data, the model is trained without active learning.
# @param data A dataset stored as a pandas DataFrame
# @param y_column The name of the column containing y values
# @param estimator The model to train. Currently use scikit-learn and keras models.
# @param limit_cols Columns in the dataset to exclude from x
# @param mode A string indicating the 'version' of the model - can be a name, or 'production'
# @return accuracy, trained model
def passive_learning(data, y_column, estimator=sklearn.ensemble.RandomForestClassifier(), limit_cols=None, mode=paths.dataset_version):
    print("training with all labelled samples")
    data = data.dropna(subset=[y_column])
    default_drop = ['DocID', 'TagMethod']
    if limit_cols:
        default_drop.extend(limit_cols)
    X, Y = mlh.data_prep(data, limit_cols=default_drop, y_column=y_column)
    if paths.production in mode:
        X_train, X_test, y_train, y_test = X, X, Y, Y  # no split test set
    else:
        test_percentage = 0.2
        print("test set size: ", test_percentage)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_percentage)
    #X, Y = X.astype(int), Y.astype(int)  # pd's Int64 dtype accepts NaN  # but Int64 dtype is "unknown"  # need to change this line to accept with str input, not sure how

    learner = estimator.fit(X_train, y_train)
    # y_pred = learner.predict(X_test)
    # accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    if 'TagMethod' in data.columns:
        valid_set = data.loc[(data['TagMethod'] == "manual") | (data['TagMethod'] == "legacy")]
    else:
        valid_set = data  # for legacy dataset
    valid_x, valid_y = mlh.data_prep(valid_set, y_column=y_column, limit_cols=default_drop)
    valid_y = valid_y.astype(int)
    # valid_x = valid_set.drop(columns=['DocID', 'TOCPage', "TagMethod"])
    y_pred = learner.predict(valid_x)
    accuracy = sklearn.metrics.accuracy_score(valid_y, y_pred)
    conf = sklearn.metrics.confusion_matrix(valid_y, y_pred)
    print("Test set results: ")
    pred = learner.predict(X_test)
    print(sklearn.metrics.accuracy_score(y_test, pred))
    print(sklearn.metrics.confusion_matrix(y_test, pred))
    print("For manually annotated:")
    print(accuracy)
    print(conf)

    # print false negatives
    # print('False negatives: ')
    # for i in range(len(valid_y)):
    #     if valid_y.iloc[i] != y_pred[i]:
    #         if y_pred[i] == 0:
    #             print(valid_x.iloc[i])


    return accuracy, learner

## Train a model
# @param data A dataset stored as a pandas DataFrame
# @param y_column The name of the column containing y values
# @param n_queries Number of points of data to label in active learning
# @param estimator The model to train. Currently use scikit-learn and keras models.
# @param datafile The filename of the dataset in 'data'
# @param limit_cols Columns in the dataset to exclude from x
# @param mode A string indicating the 'version' of the model - can be a name, or 'production'
# @return accuracy, trained model
def train(data, y_column, n_queries, estimator, datafile, limit_cols=None, mode=paths.dataset_version):
    unlabelled = data[y_column].loc[data[y_column].isnull()]

    if len(unlabelled) < n_queries:  # if less unlabelled than want to sample, reduce sample size
        if len(unlabelled) == 0:
            data[y_column].loc[data['TagMethod'] == 'auto'] = np.nan
        else:
            n_queries = len(unlabelled)

    if n_queries > 0:
        updated_data, accuracy, learner = active_learning(data, n_queries, y_column, estimator, limit_cols, mode)
        updated_data.to_csv(datafile, index=False)  # save slightly more annotated dataset
    else:
        accuracy, learner = passive_learning(data, y_column, estimator, limit_cols, mode)
    return accuracy, learner

## Adds labels to unlabelled members of a dataset using predictions from its respective model, and saves to the same file.
# @param type name of the model
# @param classification_function function inside the model which acts like predict()
# @param y_column The name of the column containing y values
# @param mode A string indicating the 'version' of the model - can be a name, or 'production'
def automatically_tag(type, classification_function, y_column, mode=paths.dataset_version):
    source = paths.get_dataset_path(type, mode)  # 'toc'
    df = pd.read_csv(source)
    df = df.reset_index(drop=True)
    new_tags = classification_function(df, masked=False) # can add mode parameter if ever use it on production set
    #idx = df.loc[((df['TagMethod'] != 'legacy') != (df['TOCPage'] == df['TOCPage'])) & (df['TagMethod'] != 'manual')].index.values #= new_tags.loc[(df['TagMethod'] != 'legacy') & (df['TagMethod'] != 'manual')]
    idx = df.loc[((df['TagMethod'] == 'auto') | (df['TagMethod'] != df['TagMethod'])) | (df[y_column] != df[y_column])].index.values  # join of auto and TOCPage==None
    df.loc[idx, y_column] = new_tags.loc[idx]
    df.loc[idx, 'TagMethod'] = 'auto'
    print(len(idx), " automatically tagged")
    #df['TagMethod'].loc[(df['TagMethod'] != 'legacy') & (df['TagMethod'] != 'manual')] = 'auto'
    if 'proba' in df.columns:
        df = df.drop(columns=['proba'])
    df.to_csv(source, index=False)

## Displays a borehole table
# @param docid Unique identifying int of report
# @param table Table number
def display_df(docid, table):
    dfs = tables.get_tables(docid)
    df = dfs[table-1]
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #display.display(df)
    print(df)
    print()
    print(docid, "table: ", table-1)
    print()

## Active learning user input loop
# @param learner The model to train. Currently use scikit-learn and keras models.
# @param inst unlabelled samples
# @param docid Unique identifying int of report
# @param n_queries number of samples
# @param classes array of ints representing classes
# @param page page number
# @param line line number
# @param mode A string indicating the 'version' of the model - can be a name, or 'production'
# @return user input class
def al_input_loop(learner, inst, docid, n_queries, classes, page=None, line=None, mode=paths.dataset_version):
    print("Waiting to display next....")
    display.clear_output(wait=True)
    #print(inst)

    pred = learner.predict(inst.reshape(1, -1))
    #preds.append(pred[0])

    if mode != 'boreholes':
        display_page(int(docid), int(page), line)  # docid, pagenum, line
    else:
        display_df(int(docid), int(page))
        #print(inst)

    time.sleep(1)  # sometimes the input box doesn't show, i think because it doesn't have the time

    print("queries: ", n_queries)
    #if i == 0:
    #    print("predict proba of all samples")
    #    print(learner.predict_proba(query_inst))
    #else:
    print("predict proba of this sample: ", learner.predict_proba([inst]))
    print("Prediction: ", pred)
    #print('Is this page a Table of Contents?')
    # print(pg_path)
    print()
    y = get_input(classes)
    return y


## Active learning data prep
# @param data A dataset stored as a pandas DataFrame
# @param y_column The name of the column containing y values
# @param limit_cols Columns in the dataset to exclude from x
# @param mode A string indicating the 'version' of the model - can be a name, or 'production'
# @return labelled and unlabelled x and y samples, and original reference to pagenums, linenums, tablenums, where applicable
def al_data_prep(data, y_column, limit_cols=None, mode=paths.dataset_version):  # to generalise further, should take limit_cols param and generalise data_prep
    default_drop = ['DocID', 'TagMethod']
    if not limit_cols:
        limit_cols = default_drop
    else:
        limit_cols.extend(default_drop)

    unlabelled = data.loc[data[y_column].isnull()]
    labelled = data.dropna(subset=[y_column])  # assume that will contain 0, 1 values
    X_initial, Y_initial = mlh.data_prep(labelled, limit_cols=limit_cols, y_column=y_column)

    refs = {}
    ref_docids = unlabelled.DocID  # removing docids from X, but keeping them around in this to ref
    refs['docids'] = ref_docids
    if y_column in ['Heading', 'Marginal']:
        ref_pagenums = unlabelled.PageNum
        refs['pagenums'] = ref_pagenums
    if y_column in ['Heading', 'Marginal']:
        ref_linenums = unlabelled.LineNum
        refs['linenums'] = ref_linenums
    if mode=='boreholes':
        refs['Tables'] = unlabelled.TableNum

    X_pool, y_pool = mlh.data_prep(unlabelled, limit_cols=limit_cols, y_column=y_column)
    ref_idx = X_pool.index.values
    refs['idx'] = ref_idx
    X_pool.dropna(inplace=True)
    #X_pool, y_pool = X_pool.to_numpy(), y_pool.to_numpy()  # COMMENT OUT FOR DEBUG
    return X_initial, Y_initial, X_pool, y_pool, refs


## Saves pages of a document as individual images. This makes the display of a page faster, as the file to be opened is much smaller.
# @param docid Unique identifying int of report
# @param report_num File number
def save_report_pages(docid, report_num=1):
    report_path = paths.get_report_name(docid, local_path=True, file_extension='.pdf', file_num=report_num)
    try:
        images = convert_from_path(report_path)
    except exceptions.PDFPageCountError:
        fname = textractor.textloading.find_file(docid)
        rep_folder = (paths.get_report_name(docid, local_path=True, file_num=report_num)).split('cr')[0]
        if not os.path.exists(rep_folder):
            os.mkdir(rep_folder)

        if '.tif' in fname:
            report_in = re.sub('.pdf', '.tif', report_path)
            textloading.download_report(fname, report_in)
            with open(report_path, "wb") as f:
                f.write(img2pdf.convert(open(report_in, "rb")))
        else:
            textloading.download_report(fname, report_path)
        images = convert_from_path(report_path)

    for i in range(len(images)):
        pgpath = paths.get_report_page_path(docid, i + 1)
        images[i].save(pgpath)


# if __name__ =="__main__":
#     #display_page('70562', 5, 4)
#     import heading_id_toc
#     automatically_tag('proc_heading_id_toc', heading_id_toc.get_toc_headings, 'Heading')

#
# if __name__ == "__main__":
#     sample = textloading.get_reportid_sample(1000, cutoffdate=None)
#     #p = subprocess.Popen([], cwd="C:/Users/andraszeka/OneDrive - ITP (Queensland Government)/gsq-boreholes/")
#     #subprocess.call("cd C:/Users/andraszeka/OneDrive - ITP (Queensland Government)/gsq-boreholes/")
#     for id in sample:
#         print("aws s3 cp s3://gsq-horizon/QDEX/" + id + " 1000sample/" + id + " --recursive")
#         #cmd = "aws s3 cp s3://gsq-horizon/QDEX/" + id + " 1000sample/" + id + "--recursive"
#         #subprocess.call(cmd)



# def data_prep(data, limit_cols=None, y_column=None):  # y=False,
#     X = data
#     if limit_cols:
#         #X = X.drop(columns=limit_cols)
#         for col in limit_cols:
#             try:
#                 X = X.drop(columns=[col])
#             except:
#                 print('column ', col, " doesn't exist in X")  # makes it ok to accidentally have multiple of the same col in limit_cols
#     if y_column:
#         X = X.drop(columns=[y_column])
#         Y = data[y_column]
#         return X, Y
#     return X



