import pandas as pd
import glob
import json
import numpy as np
import re
import settings
import sklearn
from sklearn import tree
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import machine_learning_helper as mlh
import active_learning as al
from sklearn.ensemble import RandomForestClassifier

name = 'fig'
y_column = 'FigPage'
columns = ['DocID', 'PageNum', 'MedConfidence', 'AvgConfidence', 'RangeConfidence', 'IQRConfidence','MedLineLen', 'ContainsFigWord', 'ContainsFigLn', 'FigPos', y_column, 'TagMethod']
limited_cols = ['DocID']


def write_to_dataset(pi, docid):
    docset = np.zeros((len(pi.items()), len(columns)))
    for info, i in zip(pi.items(), range(len(pi))):
        fig = 0
        figln = 0
        figlnpos = -1
        confs = []
        linelens = []
        for inf, j in zip(info[1], range(len(info[1]))):
            confs.append(inf['Confidence'])
            linelens.append(len(inf['Text']))
            if 'figure' in inf['Text'].lower():
                fig = 1
                if figlnpos == -1:
                    figlnpos = j / len(info[1])
                if re.search(r'Figure\s\d+\.*\s*\w*', inf['Text']) or re.search(r'FIGURE\s\d+\.*\s*\w*', inf['Text']):
                    figln = 1
                    figlnpos = j / len(info[1])
        medconf = np.median(np.array(confs))
        avgconf = np.average(np.array(confs))
        rangeconf = np.max(np.array(confs)) - np.min(np.array(confs))
        medlineln = np.median(np.array(linelens))

        q75, q25 = np.percentile(np.array(confs), [75, 25])
        iqr = q75 - q25

        docset[i] = np.array([docid.strip('cr_'), info[0], medconf, avgconf, rangeconf, iqr, medlineln, fig, figln, figlnpos, 0])
    return docset


def create_dataset():
    df = pd.DataFrame(columns=columns)
    pageinfos = sorted(glob.glob('training/restructpageinfo/*.json'))

    for pagesinfo in pageinfos:
        pi = json.load(open(pagesinfo))
        #docset = np.zeros((len(pi.items()), 11))
        docid = pagesinfo.split('\\')[-1].replace('_1_restructpageinfo.json', '')
        docset = write_to_dataset(pi, docid)
        pgdf = pd.DataFrame(data=docset, columns=columns)
        df = df.append(pgdf, ignore_index=True)
    return df


def create_individual_dataset(docid, docinfo, doclines):
    pi = docinfo
    docset = write_to_dataset(pi, docid)
    df = pd.DataFrame(docset, columns=columns)
#    df = df.drop(columns=['TagMethod'])  # don't want to have this and the y column?
    return df


def train(n_queries=10, mode=settings.dataset_version):  # datafile=settings.get_dataset_path(name), model_file=settings.get_model_path(name),
    datafile = settings.get_dataset_path(name, mode)
    data = pd.read_csv(datafile)
    clf = RandomForestClassifier() #tree.DecisionTreeClassifier()
    accuracy, clf = al.active_learning(data, n_queries, y_column, estimator=clf, limit_cols=limited_cols)
    print(accuracy)
    model_file = settings.get_model_path(name, mode)
    with open(model_file, "wb") as file:
        pickle.dump(clf, file)

#
# def classify_page(data, mode=settings.dataset_version):
#     if mode == settings.dataset_version:
#         if not os.path.exists(settings.get_model_path(name)):
#             train(data, n_queries=0)
#     return mlh.classify(data, name, mode=mode, limit_cols=limited_cols)
#

def get_fig_pages(data, mode=settings.dataset_version): #docid, docinfo, doclines):  # change usages of this to pass fig dataset
    return mlh.get_classified(data, name, y_column, limited_cols, mode)
    # if not docid:
    #     data_file = settings.dataset_path + 'fig_dataset.csv'
    #     df = pd.read_csv(data_file)
    # else:
    #     df = create_individual_dataset(docid, docinfo, doclines)
    # classes = classify_page(df)
    # mask = np.array([True if i==1 else False for i in classes])
    # fig_pages = df[mask]
    # return fig_pages


if __name__ == "__main__":
    #dataset = create_dataset()
    #dataset.to_csv('fig_dataset.csv', index=False)
    #print(dataset)
    matplotlib.rcParams['figure.figsize'] = (20, 20)
    data = 'fig_dataset.csv'
    data = pd.read_csv(data)
    X = data.drop(['DocID', 'FigPage'], axis=1)
    Y = data['FigPage']
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33)

    #fig_pages = get_fig_pages()
    #print(fig_pages)
