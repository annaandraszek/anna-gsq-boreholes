# finding page(s) in a document that is a table of contents

# b) classifier use
    # 1. use with real documents
        # pipeline:
            # input document name on s3 to download, or input local document name, or input json name to not run textract

import pandas as pd
import numpy as np
import glob
import settings
import json
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import sklearn
import pickle
import os
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling
from pdf2image import convert_from_path, exceptions
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
from IPython import display
import textloading
import textracting
import re
import img2pdf
import time

columns = ['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'ContainsListOf',
           'PrevPageTOC', 'TOCPage', 'TagMethod']


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


def create_dataset():
    #columns = ['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'ContainsListOf', 'PrevPageTOC', 'TOCPage']
    df = pd.DataFrame(columns=columns)
    pageinfos = sorted(glob.glob('training/restructpageinfo/*.json'))
    #pagelines = sorted(glob.glob('training/pagelines/*'))

    for pagesinfo in pageinfos:
        pi = json.load(open(pagesinfo))
        #pl = json.load(open(pageslines))
        docset = np.zeros((len(pi.items()), len(columns)))
        docid = (pagesinfo.split('\\')[-1].replace('_1_restructpageinfo.json', '')).strip('cr_')
        #save_report_pages(docid)
        for info, j in zip(pi.items(), range(len(pi.items()))):
            toc = 0
            c = 0
            listof = 0
            prev_pg_toc = 0  # indicates in the previous page is a TOC - to find second pages of this
            for line in info[1]:
                if 'contents' in line['Text'].lower():
                    c = 1
                    if 'table of contents' in line['Text'].lower():
                        toc = 1
                if 'list of' in line['Text'].lower():
                    listof = 1
                # if docset[j-1][3] == 1 or docset[j-1][4] == 1:
                #     prev_pg_toc = 1
            docset[j] = np.array([docid, info[0], len(info[1]), toc, c, listof, prev_pg_toc, None, None])
        pgdf = pd.DataFrame(data=docset, columns=columns)
        df = df.append(pgdf, ignore_index=True)

    prev_toc_dataset = settings.dataset_path + 'toc_dataset.csv'
    if os.path.exists(prev_toc_dataset):
        prev = pd.read_csv(prev_toc_dataset, dtype=int)
        df['TOCPage'] = df.apply(lambda x: assign_y(x, prev), axis=1)
        df['TagMethod'].loc[df['TOCPage'] == df['TOCPage']] = "legacy"
    return df


def assign_y(x, prev):
    d, p = int(x['DocID']), int(x['PageNum'])
    y = (prev['TOCPage'].loc[(prev['DocID'] == d) & (prev['PageNum'] == p)])
    if len(y) == 0:
        return None
    elif len(y) == 1:
        return y.values[0]
    else:
        print("more rows than 1? ")
        print(y.values)


def data_prep(data, y=False, limit_cols=None):
    #data = data.drop(['Comments'], axis=1)
    #data = data.dropna()
    X = data.drop(columns=['TOCPage', 'TagMethod'])
    #X = data[['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'ContainsListOf', 'PrevPageTOC']]
    if limit_cols:
        X = X.drop(columns=limit_cols)
    if y:
        Y = data.TOCPage
        return X, Y  #.astype(pd.Int64Dtype()), Y.astype(pd.Int64Dtype())
    return X  #.astype(pd.Int64Dtype())


def train(datafile=settings.get_dataset_path('toc'), n_queries=10):
    data = pd.read_csv(datafile)
    data[['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'ContainsListOf',
          'PrevPageTOC', 'TOCPage']] = data[['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'ContainsListOf',
           'PrevPageTOC', 'TOCPage']].astype("Int64")
    data['TagMethod'] = data['TagMethod'].astype("string")
    classes = [0, 1, 2]

    unlabelled = data.loc[data['TOCPage'].isnull()]
    labelled = data.dropna(subset=['TOCPage'])  # assume that will contain 0, 1 values

    if n_queries=='all':
        n_queries = unlabelled.shape[0]

    elif isinstance(n_queries, int) & (len(unlabelled) < n_queries):  # if less unlabelled than want to sample, reduce sample size
        n_queries = len(unlabelled)

    if n_queries > 0:
        X_initial, Y_initial = data_prep(labelled, y=True, limit_cols=['DocID'])

        ref_docids = unlabelled.DocID  # removing docids from X, but keeping them around in this to ref
        X_pool, y_pool = data_prep(unlabelled, y=True, limit_cols=['DocID'])
        ref_idx = X_pool.index.values
        X_pool, y_pool = X_pool.to_numpy(), y_pool.to_numpy()

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_initial, Y_initial, test_size = 0.90)

        learner = ActiveLearner(estimator=RandomForestClassifier(),
                                query_strategy=uncertainty_sampling,
                                X_training=X_train, y_training=y_train.astype(int))

        accuracy_scores = [learner.score(X_test, y_test.astype(int))]
        preds = []
        #anno_df = data.copy(deep=True)
        query_idx, query_inst = learner.query(X_pool, n_instances=n_queries)
        y_new = np.zeros(n_queries, dtype=int)
        for i in range(n_queries):
            print("Waiting to display next page....")
            display.clear_output(wait=True)
            pred = learner.predict(query_inst[i].reshape(1, -1))
            preds.append(pred[0])
            display_page(int(ref_docids.iloc[query_idx[i]]), int(query_inst[i][0]))  #docid, pagenum
            time.sleep(1)  # sometimes the input box doesn't show, i think because it doesn't have the time

            print("queries: ", n_queries)
            if i == 0:
                print("predict proba of all samples")
                print(learner.predict_proba(query_inst))
            else:
                print("predict proba of this sample: ", learner.predict_proba([query_inst[i]]))
            print("Prediction: ", pred)
            print('Is this page a Table of Contents?')
            #print(pg_path)
            y = -1
            while y not in classes:
                print("Enter one of: ", str(classes))
                y = input()
                try:
                    y = int(y)  # set it as int here instead of on input to avoid error breaking execution when input is bad
                except:
                    continue
            y_new[i] = y
            print()
            data.at[ref_idx[query_idx[i]], 'TOCPage'] = y  # save value to copy of data
            data.at[ref_idx[query_idx[i]], 'TagMethod'] = 'manual'

        learner.teach(query_inst, y_new)  # reshape 1, -1
        accuracy_scores.append(learner.score(X_test, y_test.astype(int)))
        print("End of annotation. Samples, predictions, annotations: ")
        print(ref_docids.iloc[query_idx].values, np.concatenate((query_inst, np.array([preds]).T, y_new.reshape(-1, 1)), axis=1))

        accuracy = accuracy_scores[-1]
        data.to_csv(datafile, index=False)  # save slightly more annotated dataset

    else:
        print("no unlabelled samples, training normally")
        data = data.dropna(subset=['TOCPage'])
        X, Y = data_prep(data, y=True, limit_cols=['DocID'])
        X, Y = X.astype(int), Y.astype(int)  # pd's Int64 dtype accepts NaN
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.20)

        learner = RandomForestClassifier() #tree.DecisionTreeClassifier()
        learner = learner.fit(X_train, y_train)
        y_pred = learner.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        print(accuracy)
    with open(settings.get_model_path('toc'), "wb") as file:
        pickle.dump(learner, file)
    print("End of training stage. Re-run to train again")
    return accuracy


def tag_prevpagetoc():
    source = settings.get_dataset_path('toc')
    df = pd.read_csv(source)
    tocdf = df.loc[df['TOCPage'] == 1]
    count = 0
    for i, row in tocdf.iterrows():
        d, p = row.DocID, row.PageNum + 1
        #try:
        view = df[(df['DocID'] == d) & (df['PageNum'] == p)]
        idx = view.index.values  # should only be one indice
        if len(idx) == 0:
            continue
        i = idx[0]
            #print("len idx: ", len(idx))  # debug check
            #for i in idx:
        if df.at[i, 'PrevPageTOC'] != 1:
            df.at[i, 'PrevPageTOC'] = 1
            count += 1
            if (df.at[i, 'TagMethod'] == 'auto') & (df.at[i, 'TOCPage'] == 0):  # only overwrite automatically tagged # if page already isn't 0, don't None
                df.at[i, 'TOCPage'] = None  # reset to tag again
        #except IndexError:  # there is no next page
        #    continue
    df.to_csv(settings.get_dataset_path('toc'), index=False)
    return count


def display_page(docid, page):
    pg_path = settings.get_report_page_path(int(docid), int(page))  # docid, page
    image = Image.open(pg_path)
    display.display(image)
    print(pg_path)


def automatically_tag():
    source = settings.get_dataset_path('toc')
    df = pd.read_csv(source)
    df = df.reset_index(drop=True)
    new_tags = classify_page(df)
    #idx = df.loc[((df['TagMethod'] != 'legacy') != (df['TOCPage'] == df['TOCPage'])) & (df['TagMethod'] != 'manual')].index.values #= new_tags.loc[(df['TagMethod'] != 'legacy') & (df['TagMethod'] != 'manual')]
    idx = df.loc[(df['TagMethod'] == 'auto') | (df['TOCPage'] != df['TOCPage'])].index.values  # join of auto and TOCPage==None
    df.loc[idx, 'TOCPage'] = new_tags[idx]
    df.loc[idx, 'TagMethod'] = 'auto'
    print(len(idx), " automatically tagged")
    #df['TagMethod'].loc[(df['TagMethod'] != 'legacy') & (df['TagMethod'] != 'manual')] = 'auto'
    df.to_csv(settings.get_dataset_path('toc'), index=False)


def check_tags(show=False):
    source = settings.get_dataset_path('toc')
    df = pd.read_csv(source)
    test = df.loc[(df['ContainsContentsWord'] == 1) | (df['ContainsTOCPhrase'] == 1) | (df['ContainsListOf'] == 1)]
    bad = test.loc[test['TOCPage'] == 0]
    print("Samples with 'Contents' or 'Table of Contents' or 'List of' but not tagged as TOC, or ListOf: ")
    c = 0
    for i, row in bad.iterrows():
        if show:
            print(i, row)
            display_page(row.DocID, row.PageNum)
        if df.at[i, 'TagMethod'] != 'manual':  # can edit auto or legacy
            df.at[i, 'TOCPage'] = None
            c += 1
    print("count: ", c)
    df.to_csv(source, index=False)


def classify_page(data):
    if not os.path.exists(settings.get_model_path('toc')):
        train(data)
    with open(settings.get_model_path('toc'), "rb") as file:
        model = pickle.load(file)
    data = data_prep(data, limit_cols=['DocID'])
    pred = model.predict(data)
    return pred


def get_toc_pages(df):
    #data_file = settings.production_path + docid + '_toc_dataset.csv'
    #df = pd.read_csv(data_file)
    classes = classify_page(df)
    mask = np.array([True if i==1 else False for i in classes])
    toc_pages = df[mask]
    return toc_pages


if __name__ == "__main__":
    #reports = get_reportid_sample()
    # rfs = glob.glob('training/QDEX/*')
    # report_ids = set([r.rsplit('\\')[-1] for r in rfs])
    # #reports = []
    # #download_reports(reports, 'training/')
    #
    # for report in report_ids:
    #     res = upload_to_my_bucket(report)
    #     if res != 0:
    #         continue
    #     doc2data(report)

    # dataset = create_dataset()
    # dataset.to_csv('toc_dataset.csv', index=False)
    # print(dataset)

    #toc_pages = get_toc_pages()
    #print(toc_pages)
    #save_report_pages('4412')
    # train()
    automatically_tag()
    # check_tags()
    # train(all=True)
    # tag_prevpagetoc()
    # train(all=True)

