import time
from IPython import display
import settings
from PIL import Image, ImageDraw
import textloading
import textracting
import re
import img2pdf
from pdf2image import convert_from_path, exceptions
import os
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json
import sklearn


def data_prep(data, limit_cols=None, y_column=None):  # y=False,
    X = data
    if limit_cols:
        X = X.drop(columns=limit_cols)
    if y_column:
        X = X.drop(columns=[y_column])
        Y = data[y_column]
        return X, Y
    return X


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


def display_page(docid, page, line=None):
    pg_path = settings.get_report_page_path(int(docid), int(page))  # docid, page
    image = Image.open(pg_path)

    if line:
        width, height = image.size

        docinfofile = settings.get_restructpageinfo_file(docid)
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

        elif bottom > height:
            change = bottom - height
            bottom = height
            top += change

        draw = ImageDraw.Draw(image, 'RGBA')
        draw.rectangle([ln_left, ln_top, ln_left + (width * box['Width']), ln_top + (height * box['Height'])], outline='green')

        crop_image = image.crop((left, top, right, bottom))
        #crop_ln_top = crop_height * box['Top']

        #draw.rectangle([ln_left, crop_ln_top, ln_left + (width * box['Width']), crop_ln_top + (crop_height * box['Height'])], outline='green')
        image = crop_image

    display.display(image)
    # line option: draw a box around the line
    # get docinfo, query the line number and bounding box
    # crop page to about 1/3 of it to make it more focused on the line

    print(pg_path)


def active_learning(data, n_queries, y_column, estimator=RandomForestClassifier, limit_cols=None):
    line = False
    if y_column in ['Marginal', 'Heading']:  # covers marginal_lines, heading_id_toc, heading_id_intext
        line = True  # determines if a line or page is to to be displayed
    classes = set(data[y_column].values)  #todo: check type

    X_initial, Y_initial, X_pool, y_pool, ref_docids, ref_idx = al_data_prep(data, y_column, limit_cols)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_initial, Y_initial,
                                                                                test_size=0.90)
    learner = ActiveLearner(estimator=estimator(), #ensemble.RandomForestClassifier(),
                            query_strategy=uncertainty_sampling,
                            X_training=X_train, y_training=y_train.astype(int))
    accuracy_scores = [learner.score(X_test, y_test.astype(int))]
    preds = []
    query_idx, query_inst = learner.query(X_pool, n_instances=n_queries)
    y_new = np.zeros(n_queries, dtype=int)
    for i in range(n_queries):
        idx = query_idx[i]
        y = al_input_loop(learner, query_inst[i], ref_docids.iloc[idx], n_queries, classes, line=line)
        y_new[i] = y
        data.at[ref_idx[idx], y_column] = y  # save value to copy of data
        data.at[ref_idx[idx], 'TagMethod'] = 'manual'

    learner.teach(query_inst, y_new)  # reshape 1, -1
    accuracy_scores.append(learner.score(X_test, y_test.astype(int)))
    #print("End of annotation. Samples, predictions, annotations: ")
    #print(ref_docids.iloc[query_idx].values,
    #      np.concatenate((query_inst, np.array([preds]).T, y_new.reshape(-1, 1)), axis=1))

    accuracy = accuracy_scores[-1]
    return data, accuracy, learner


def passive_learning(data, y_column, estimator=sklearn.ensemble.RandomForestClassifier(), limit_cols=None):
    print("training with all labelled samples")
    data = data.dropna(subset=[y_column])
    default_drop = ['DocID', 'TagMethod']
    if limit_cols:
        default_drop.extend(limit_cols)
    X, Y = data_prep(data, limit_cols=default_drop, y_column=y_column)
    X, Y = X.astype(int), Y.astype(int)  # pd's Int64 dtype accepts NaN  # but Int64 dtype is "unknown"
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.20)

    learner = estimator.fit(X_train, y_train)
    # y_pred = learner.predict(X_test)
    # accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    valid_set = data.loc[(data['TagMethod'] == "manual") | (data['TagMethod'] == "legacy")]
    valid_x, valid_y = data_prep(valid_set, y_column=y_column, limit_cols=['DocID', 'TagMethod'])
    valid_y = valid_y.astype(int)
    # valid_x = valid_set.drop(columns=['DocID', 'TOCPage', "TagMethod"])
    y_pred = learner.predict(valid_x)
    accuracy = sklearn.metrics.accuracy_score(valid_y, y_pred)
    conf = sklearn.metrics.confusion_matrix(valid_y, y_pred)
    print(accuracy)
    print(conf)
    return accuracy, learner


def train(data, y_column, n_queries, estimator, datafile, limit_cols=None):
    unlabelled = data[y_column].loc[data[y_column].isnull()]

    if len(unlabelled) < n_queries:  # if less unlabelled than want to sample, reduce sample size
        n_queries = len(unlabelled)

    if n_queries > 0:
        updated_data, accuracy, learner = active_learning(data, n_queries, y_column, estimator, limit_cols)
        updated_data.to_csv(datafile, index=False)  # save slightly more annotated dataset
    else:
        accuracy, learner = passive_learning(data, y_column, estimator, limit_cols)
    return accuracy, learner


def automatically_tag(type, classification_function, y_column):
    source = settings.get_dataset_path(type)  # 'toc'
    df = pd.read_csv(source)
    df = df.reset_index(drop=True)
    new_tags = classification_function(df) #classify_page(df)
    #idx = df.loc[((df['TagMethod'] != 'legacy') != (df['TOCPage'] == df['TOCPage'])) & (df['TagMethod'] != 'manual')].index.values #= new_tags.loc[(df['TagMethod'] != 'legacy') & (df['TagMethod'] != 'manual')]
    idx = df.loc[(df['TagMethod'] == 'auto') | (df[y_column] != df[y_column])].index.values  # join of auto and TOCPage==None
    df.loc[idx, y_column] = new_tags[idx]
    df.loc[idx, 'TagMethod'] = 'auto'
    print(len(idx), " automatically tagged")
    #df['TagMethod'].loc[(df['TagMethod'] != 'legacy') & (df['TagMethod'] != 'manual')] = 'auto'
    df.to_csv(settings.get_dataset_path(type), index=False)


def al_input_loop(learner, inst, docid, n_queries, classes, line=None):
    print("Waiting to display next....")
    display.clear_output(wait=True)
    pred = learner.predict(inst.reshape(1, -1))
    #preds.append(pred[0])
    if line:
        line = int(inst[1])
    display_page(int(docid), int(inst[0]), line)  # docid, pagenum, line

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


def al_data_prep(data, y_column, limit_cols=None):  # to generalise further, should take limit_cols param and generalise data_prep
    default_drop = ['DocID', 'TagMethod']
    if not limit_cols:
        limit_cols = default_drop
    else:
        limit_cols.extend(default_drop)

    unlabelled = data.loc[data[y_column].isnull()]
    labelled = data.dropna(subset=[y_column])  # assume that will contain 0, 1 values
    X_initial, Y_initial = data_prep(labelled, limit_cols=limit_cols, y_column=y_column)

    ref_docids = unlabelled.DocID  # removing docids from X, but keeping them around in this to ref
    X_pool, y_pool = data_prep(unlabelled, limit_cols=limit_cols, y_column=y_column)
    ref_idx = X_pool.index.values
    X_pool, y_pool = X_pool.to_numpy(), y_pool.to_numpy()
    return X_initial, Y_initial, X_pool, y_pool, ref_docids, ref_idx


def save_report_pages(docid):
    report_path = settings.get_report_name(docid, local_path=True, file_extension='.pdf')
    try:
        images = convert_from_path(report_path)
    except exceptions.PDFPageCountError:
        fname = textracting.find_file(docid)
        rep_folder = (settings.get_report_name(docid, local_path=True)).split('cr')[0]
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
        pgpath = settings.get_report_page_path(docid, i+1)
        images[i].save(pgpath)


if __name__ =="__main__":
    display_page('70562', 5, 4)