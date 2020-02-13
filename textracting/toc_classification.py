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
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import active_learning
import machine_learning_helper as mlh


y_column = 'TOCPage'
columns = ['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'ContainsListOf',
           'PrevPageTOC', y_column, 'TagMethod']
estimator = RandomForestClassifier()

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
    #y_column = 'TOCPage'
    df = mlh.add_legacy_y(prev_toc_dataset, df, y_column)
    return df


def train(datafile=settings.get_dataset_path('toc'), n_queries=10):
    data = pd.read_csv(datafile)
    data[['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'ContainsListOf',
          'PrevPageTOC', 'TOCPage']] = data[['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'ContainsListOf',
           'PrevPageTOC', 'TOCPage']].astype("Int64")
    data['TagMethod'] = data['TagMethod'].astype("string")
    accuracy, learner = active_learning.train(data, y_column, n_queries, estimator, datafile)
    with open(settings.get_model_path('toc'), "wb") as file:
        pickle.dump(learner, file)
    print("End of training stage. Re-run to train again")
    return accuracy


def tag_prevpagetoc():
    source = settings.get_dataset_path('toc')
    df = pd.read_csv(source)
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
    df.to_csv(settings.get_dataset_path('toc'), index=False)
    return count


def check_tags(show=False):
    source = settings.get_dataset_path('toc')
    df = pd.read_csv(source)
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
    df.to_csv(source, index=False)


def classify_page(data, mode=settings.dataset_version):
    if mode == settings.dataset_version:
        if not os.path.exists(settings.get_model_path('toc')):
            train(data)
    return mlh.classify(data, 'toc', mode=mode, limit_cols=['DocID'])


def get_toc_pages(df, mode=settings.dataset_version):
    #data_file = settings.production_path + docid + '_toc_dataset.csv'
    #df = pd.read_csv(data_file)
    classes = classify_page(df, mode)  #classify_page(df)
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
    train(n_queries=5)
    #automatically_tag()
    # check_tags()
    # train(all=True)
    # tag_prevpagetoc()
    # train(all=True)

