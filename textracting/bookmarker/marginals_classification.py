## @file
# Module functions for classifying a line as a marginal (header or footer)

import numpy as np
import pandas as pd
import json
import glob
import settings
from sklearn import ensemble
import pickle
import re
from bookmarker import active_learning, machine_learning_helper as mlh


name = 'marginal_lines'
y_column = 'Marginal'
columns = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum','Text', 'Words2Width', 'WordsWidth', 'Width', 'Height',
           'Left', 'Top', 'ContainsNum', 'ContainsTab', 'ContainsPage', 'Centrality', y_column, 'TagMethod']
limit_cols=['DocID', 'Text', 'LineNum']
include_cols = ['PageNum', 'NormedLineNum', 'Words2Width', 'WordsWidth', 'Width', 'Height', 'Left', 'Top',
                'ContainsNum', 'ContainsTab', 'ContainsPage', 'Centrality']
estimator = ensemble.RandomForestClassifier()
data_path = settings.get_dataset_path(name)
model_path = settings.get_model_path(name)


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


def write_to_dataset(df, pi, docid):
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
            # normed_line = line['LineNum'] / lines
            centrality = 0.5 - abs(bb['Left'] + (bb['Width'] / 2) - 0.5)  # the higher value the more central
            words2width = line['WordsWidth'] / bb['Width']
            docset.append([docid, int(page), line['LineNum'], 0, line['Text'], words2width, line['WordsWidth'],
                           bb['Width'], bb['Height'], bb['Left'], bb['Top'], contains_num, contains_tab,
                           contains_page, centrality, None, None])

        temp = pd.DataFrame(data=docset, columns=columns)
        if (max(temp['LineNum']) - min(temp['LineNum'])) == 0:  # only one line # avoid NaN from div by 0
            temp['NormedLineNum'] = 0
        else:
            temp['NormedLineNum'] = (temp['LineNum'] - min(temp['LineNum'])) / (
                        max(temp['LineNum']) - min(temp['LineNum']))
        df = df.append(temp, ignore_index=True)
    return df


def create_dataset():
    pageinfos = glob.glob('training/restructpageinfo/*.json')
    df = pd.DataFrame(columns=columns)
    for pagesinfo in pageinfos:
        pi = json.load(open(pagesinfo))
        docid = pagesinfo.split('\\')[-1].replace('_1_restructpageinfo.json', '').strip('cr_')
        write_to_dataset(df, pi, docid)

    unnormed = np.array(df['Centrality'])
    normalized = (unnormed - min(unnormed)) / (max(unnormed) - min(unnormed))
    df['Centrality'] = normalized

    prev_dataset = settings.dataset_path + 'marginals_dataset_v2.csv'
    df = mlh.add_legacy_y(prev_dataset, df, y_column, line=True)
    return df


def create_individual_dataset(docid):
    pageinfo = settings.get_restructpageinfo_file(docid)
    pi = json.load(open(pageinfo))
    df = pd.DataFrame(columns=columns)
    write_to_dataset(df, pi, docid)
    unnormed = np.array(df['Centrality'])
    normalized = (unnormed - min(unnormed)) / (max(unnormed) - min(unnormed))
    df['Centrality'] = normalized
    return df


def train(n_queries=10, mode=settings.dataset_version): #, model='forest') datafile=data_path,
    datafile = settings.get_dataset_path(name, mode)  # need to define these here because mode may be production
    model_path = settings.get_model_path(name, mode)
    data = pd.read_csv(datafile)
    accuracy, learner = active_learning.train(data, y_column, n_queries, estimator, datafile, limit_cols=limit_cols)
    with open(model_path, "wb") as file:
        pickle.dump(learner, file)
    print("End of training stage. Re-run to train again")
    return accuracy


def get_marginals(data, mode=settings.dataset_version):
    return mlh.get_classified(data, name, y_column, limit_cols, mode)



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
    #edit_dataset()
    train()