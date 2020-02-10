import numpy as np
import pandas as pd
import json
import glob
import settings
from sklearn import ensemble
import pickle
import os
import re
import active_learning


def contains_num(string):
    if re.search(r'(\s|^)[0-9]+(\s|$)', string):
        return 1
    return 0


def contains_tab(string):
    if re.search(r'\t', string):
        return 1
    return 0


def contains_page(string):
    if 'page' in string.lower():
        return 1
    return 0


columns = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum','Text', 'Words2Width', 'WordsWidth', 'Width', 'Height',
           'Left', 'Top', 'ContainsNum', 'ContainsTab', 'ContainsPage', 'Centrality', 'Marginal', 'TagMethod']


def assign_y(x, prev):
    d, p, l = int(x['DocID']), int(x['PageNum']), int(x['LineNum'])
    y = (prev['Marginal'].loc[(prev['DocID'] == d) & (prev['PageNum'] == p) & (prev['LineNum'] == l)])
    if len(y) == 0:
        return None
    elif len(y) == 1:
        return y.values[0]
    else:
        print("more rows than 1? ")
        print(y.values)


def create_dataset():
    pageinfos = glob.glob('training/restructpageinfo/*.json')
    df = pd.DataFrame(columns=columns)
    for pagesinfo in pageinfos:
        pi = json.load(open(pagesinfo))
        docid = pagesinfo.split('\\')[-1].replace('_1_restructpageinfo.json', '').strip('cr_')
        for info in pi.items():
            docset = []
            page = info[0]
            for line in info[1]:
                contains_num = 0
                contains_tab = 0
                contains_page = 0
                bb = line['BoundingBox']
                if re.search(r'(\s|^)[0-9]+(\s|$)', line['Text']):
                    contains_num = 1
                if re.search(r'\t', line['Text']):
                    contains_tab = 1
                if 'page' in line['Text'].lower():
                    contains_page = 1
                #normed_line = line['LineNum'] / lines
                centrality = 0.5 - abs(bb['Left'] + (bb['Width']/2) - 0.5)  # the higher value the more central
                words2width = line['WordsWidth'] / bb['Width']
                docset.append([docid, int(page), line['LineNum'], 0, line['Text'], words2width, line['WordsWidth'],
                               bb['Width'], bb['Height'], bb['Left'], bb['Top'], contains_num, contains_tab,
                               contains_page, centrality, None, None])

            temp = pd.DataFrame(data=docset, columns=columns)
            if (max(temp['LineNum']) - min(temp['LineNum'])) == 0:  # only one line # avoid NaN from div by 0
                temp['NormedLineNum'] = 0
            else:
                temp['NormedLineNum'] = (temp['LineNum'] - min(temp['LineNum'])) / (max(temp['LineNum']) - min(temp['LineNum']))
            df = df.append(temp, ignore_index=True)

    unnormed = np.array(df['Centrality'])
    normalized = (unnormed - min(unnormed)) / (max(unnormed) - min(unnormed))
    df['Centrality'] = normalized
    df['Marginal'] = 0

    prev_dataset = settings.dataset_path + 'marginals_dataset_v2.csv'
    if os.path.exists(prev_dataset):
        prev = pd.read_csv(prev_dataset)
        df['Marginal'] = df.apply(lambda x: assign_y(x, prev), axis=1)
        df['Marginal'].loc[df['Marginal'] == 2] = 1  # removing the [2] class
        df['TagMethod'].loc[df['Marginal'] == df['Marginal']] = "legacy"

    return df


def create_individual_dataset(docid):
    pageinfo = settings.get_restructpageinfo_file(docid)
    pi = json.load(open(pageinfo))
    df = pd.DataFrame(columns=columns)
    for info in pi.items():
        docset = []
        page = info[0]
        for line in info[1]:
            contains_num = 0
            contains_tab = 0
            contains_page = 0
            bb = line['BoundingBox']
            if re.search(r'(\s|^)[0-9]+(\s|$)', line['Text']):
                contains_num = 1
            if re.search(r'\t', line['Text']):
                contains_tab = 1
            if 'page' in line['Text'].lower():
                contains_page = 1
            centrality = 0.5 - abs(bb['Left'] + (bb['Width']/2) - 0.5)  # the higher value the more central
            words2width = line['WordsWidth'] / bb['Width']
            docset.append([docid, int(page), line['LineNum'], 0, line['Text'], words2width, line['WordsWidth'],
                           bb['Width'], bb['Height'], bb['Left'], bb['Top'], contains_num, contains_tab,
                           contains_page, centrality, 0])

        temp = pd.DataFrame(data=docset, columns=columns)
        temp['NormedLineNum'] = (temp['LineNum'] - min(temp['LineNum'])) / (max(temp['LineNum']) - min(temp['LineNum']))
        df = df.append(temp, ignore_index=True)

    unnormed = np.array(df['Centrality'])
    normalized = (unnormed - min(unnormed)) / (max(unnormed) - min(unnormed))
    df['Centrality'] = normalized
    return df


def edit_dataset():
    file = settings.get_dataset_path('marginal_lines')
    data = pd.read_csv(file)
    data['Marginal'].loc[data['Marginal'] == 2] = 1
#     data['NormedLineNum'].loc[data['NormedLineNum'] != data['NormedLineNum']] = 0  # values where NormedLineNum == None
    data.to_csv(file, index=False)


def data_prep(data, y=False, limit_cols=None):
    #data.dropna(inplace=True)
    data.drop(data[data['Width'] < 0].index, inplace=True)
    X = data.drop(columns=['Text', 'Marginal', 'TagMethod'])  # PageNum?
    #X = X.drop(['Words2Width'], axis=1)  # temporarily
    if limit_cols:
        X = X.drop(columns=limit_cols)
    if y:
        Y = data['Marginal']
        return X, Y
    return X

import time


def train(datafile=settings.get_dataset_path('marginal_lines'), n_queries=10): #, model='forest'):
    data = pd.read_csv(datafile)
    y_column = 'Marginal'
    estimator = ensemble.RandomForestClassifier()
    accuracy, learner = active_learning.train(data, y_column, n_queries, estimator, datafile, limit_cols=['Text'])
    with open(settings.get_model_path('marginal_lines'), "wb") as file:
        pickle.dump(learner, file)
    print("End of training stage. Re-run to train again")
    return accuracy


def classify(data):
    if not os.path.exists(settings.get_model_path('marginal_lines')):
        train(data)
    with open(settings.get_model_path('marginal_lines'), "rb") as file:
        model = pickle.load(file)
    data = data_prep(data)
    pred = model.predict(data)
    return pred


def get_marginals(data):
    #data = create_individual_dataset(docid)
    result = classify(data)
    data['Marginal'] = result
    marginals = data.loc[data['Marginal'] != 0]
    return marginals


if __name__ == "__main__":
    #df = create_dataset()
    #df.to_csv(settings.dataset_path + 'marginals_dataset_v2.csv', index=False)

    #matplotlib.rcParams['figure.figsize'] = (20, 20)
    #file = 'marginals_dataset_v2.csv'
    #data = pd.read_csv(settings.dataset_path + file)
    #get_marginals(data)
    #train(data, model='forest')

    #classes = classify(data)
    #data['Marginal'] = classes
    #data.to_csv(settings.dataset_path + 'marginals_dataset_v2_tagged.csv', index=False)
    edit_dataset()
    #train()