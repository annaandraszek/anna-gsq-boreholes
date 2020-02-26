## @file
# finding page(s) in a document that is a table of contents

# b) classifier use
    # 1. use with real documents
        # pipeline:
            # input document name on s3 to download, or input local document name, or input json name to not run textract

## @file
# Module functions for classifying a page as a Table of Contents

import pandas as pd
import numpy as np
import glob
import settings
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pickle
import os
import active_learning
import machine_learning_helper as mlh
import matplotlib.pyplot as plt


name = 'toc'
y_column = 'TOCPage'
columns = ['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'ContainsListOf',
           'PrevPageTOC', y_column, 'TagMethod']
limit_cols = ['DocID']
include_cols = ['PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'ContainsListOf', 'PrevPageTOC']
estimator = tree.DecisionTreeClassifier()
#estimator = RandomForestClassifier()
model_path = settings.get_model_path(name)
data_path = settings.get_dataset_path(name)


def create_dataset():
    df = pd.DataFrame(columns=columns)
    pageinfos = sorted(glob.glob('training/restructpageinfo/*.json'))

    for pagesinfo in pageinfos:
        pi = json.load(open(pagesinfo))
        docset = np.zeros((len(pi.items()), len(columns)))
        docid = (pagesinfo.split('\\')[-1].replace('_1_restructpageinfo.json', '')).strip('cr_')
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
            docset[j] = np.array([docid, info[0], len(info[1]), toc, c, listof, prev_pg_toc, None, None])
        pgdf = pd.DataFrame(data=docset, columns=columns)
        df = df.append(pgdf, ignore_index=True)

    prev_toc_dataset = settings.dataset_path + 'toc_dataset.csv'
    df = mlh.add_legacy_y(prev_toc_dataset, df, y_column)
    return df


def train(n_queries=10, mode=settings.dataset_version): #datafile=data_path, ):
    datafile = settings.get_dataset_path(name, mode)
    data = pd.read_csv(datafile)
    accuracy, learner = active_learning.train(data, y_column, n_queries, estimator, datafile, mode=mode)
    if type(learner) == tree._classes.DecisionTreeClassifier:
        tree.plot_tree(learner, feature_names=include_cols, class_names=True)
        plt.show()
    with open(settings.get_model_path(name, mode), "wb") as file:
        pickle.dump(learner, file)
    print("End of training stage. Re-run to train again")
    return accuracy


def tag_prevpagetoc():
    #source = settings.get_dataset_path(name)
    df = pd.read_csv(data_path)
    tocdf = df.loc[df[y_column] == 1]
    count = 0
    for i, row in tocdf.iterrows():
        d, p = row.DocID, row.PageNum + 1
        view = df[(df['DocID'] == d) & (df['PageNum'] == p)]
        idx = view.index.values  # should only be one indice
        if len(idx) == 0:
            continue
        i = idx[0]
        if df.at[i, 'PrevPageTOC'] != 1:
            df.at[i, 'PrevPageTOC'] = 1
            count += 1
            if (df.at[i, 'TagMethod'] == 'auto') & (df.at[i, y_column] == 0):  # only overwrite automatically tagged # if page already isn't 0, don't None
                df.at[i, y_column] = None  # reset to tag again
    df.to_csv(data_path, index=False)
    return count


def check_tags(show=False):
    #source = settings.get_dataset_path(name)
    df = pd.read_csv(data_path)
    test = df.loc[(df['ContainsContentsWord'] == 1) | (df['ContainsTOCPhrase'] == 1) | (df['ContainsListOf'] == 1)]
    bad = test.loc[test[y_column] == 0]
    print("Samples with 'Contents' or 'Table of Contents' or 'List of' but not tagged as TOC, or ListOf: ")
    c = 0
    for i, row in bad.iterrows():
        if show:
            print(i, row)
            active_learning.display_page(row.DocID, row.PageNum)
        if df.at[i, 'TagMethod'] != 'manual':  # can edit auto or legacy
            df.at[i, y_column] = None
            c += 1
    print("count: ", c)
    df.to_csv(data_path, index=False)


def get_toc_pages(df, mode=settings.dataset_version):
    prevpgtoc = False
    res = []
    for i, row in df.iterrows():
        if prevpgtoc:
            row['PrevPageTOC'] = 1
        pred = mlh.get_classified(pd.DataFrame(data=[row], columns=df.columns), name, y_column, limit_cols, mode, masked=False)
        if pred[y_column].iloc[0] == 1:
            prevpgtoc = True
        else:
            prevpgtoc = False
        res.append(pred.values[0])
    res_df = pd.DataFrame(data=res, columns=pred.columns)
    return res_df.loc[res_df[y_column] == 1]
    #return mlh.get_classified(df, name, y_column, limit_cols, mode)



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
    train(n_queries=0, mode=settings.production)
    #automatically_tag()
    # check_tags()
    # train(all=True)
    # tag_prevpagetoc()
    # train(all=True)

