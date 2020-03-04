import pandas as pd
import settings
from io import StringIO
import glob
from textracting import texttransforming
import json
import pandas.errors
import os
import re
import numpy as np
from bookmarker import active_learning
import pickle
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# extract tables from a csv
def get_tables(docid):
    tablefile = settings.get_tables_file(docid)
    with open(tablefile, "r") as f:
        raw_tables = f.read()

    tables = []
    table = ""
    prev_ln_table = 0
    lines = raw_tables.split('\n')
    for line in lines:
        if not line:
            if len(table) > 0 and not prev_ln_table:
                tables.append(table)
                table = ""
        else:
            if 'Table: Table' in line:
                prev_ln_table = 1
            else:
                prev_ln_table = 0
                table += line+'\n'

    dfs = []
    for table in tables:
        data = StringIO(table)
        try:
            df = pd.read_csv(data, sep='`')
        except pandas.errors.ParserError:
            continue
        df.dropna(axis=1, how="all", inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        #df = pd.DataFrame(table)
        #print(df.columns.values)
        dfs.append(df)
    return dfs


def create_dataset():
    dataset = settings.get_dataset_path('tables', 'boreholes')
    #docids = ['32730', '44448', '37802', '2646', '44603']
    docids = []
    lines_docs = glob.glob('training/fulljson/*.json')
    for lines_doc in lines_docs:
        docid = int(lines_doc.split('\\')[-1].replace('_1_fulljson.json', '').strip('cr_'))
        docids.append(docid)
    cols = ['DocID', 'TableNum', 'Columns']
    all_columns = pd.DataFrame(columns=cols)
    for id in docids:
        # try:
        #     texttransforming.save_tables_and_kvs(id)
        # except json.decoder.JSONDecodeError:
        #     print(id)
        #     continue
        tables = get_tables(id)
        #columns = pd.Series([table.columns.values for table in tables])
        tables_values = [list(table.columns.values) for table in tables]
        exclude = ['Unnamed: ', 'nan']
        for t, i in zip(tables, range(len(tables))):
            for j, row in t.iterrows():
                tables_values[i] = np.concatenate((tables_values[i], row.values))
                tables_values[i] = [v for v in tables_values[i] if re.match(r'[A-z]+', str(v))]
                #tables_values[i] = [v for v in tables_values[i] if 'Unnamed:' not in str(v)]
                #tables_values[i] = [v for v in tables_values[i] if str(v) != 'nan']
        tables_values = pd.Series(tables_values)
        docids = pd.Series([id for x in range(len(tables_values))])
        tablenums = pd.Series([x + 1 for x in range(len(tables_values))])
        series = [docids, tablenums, tables_values]
        iddf = pd.concat(series, axis=1)
        iddf.columns = cols
        #all_columns = all_columns.append(pd.Series(columns), ignore_index=True)
        all_columns = all_columns.append(iddf, ignore_index=True)
    all_columns.to_csv(dataset, index=False)
    print('Done creating ', dataset)


def list2str(lst):
    #table = ""
    #for e in lst:
    #    table += str(e) + ' '
    return lst[0]


def concat_tables(df):
    if isinstance(df, np.ndarray): #or isinstance(df, list):
        if len(df.shape) > 1:
            return df[:, 0]
        return df
    if isinstance(df, list):
        if isinstance(df[0], list) or isinstance(df[0], np.ndarray):
            #if len(df.shape) > 1:
            return df[0]
        return df
    if isinstance(df, pd.DataFrame):
        if len(df.shape) > 1:
            #if df.shape[0]
            return df.iloc[:, 0]
        return df
    #series = serie.apply(lambda x: list2str(x))


def train(n_queries=10):
    datafile = settings.get_dataset_path('tables', 'boreholes')
    df = pd.read_csv(datafile)
    df = df.loc[df['Columns'] != '[]']
    y_column = 'Class'
    limit_cols = ['DocID', 'TableNum']
    clf = Pipeline([
        ('list2str', FunctionTransformer(concat_tables)),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 4))),
        ('cnb', ComplementNB(norm=True))
    ], verbose=True)
    accuracy, learner = active_learning.train(df, y_column, n_queries, clf, datafile, limit_cols=limit_cols,
                                              mode='boreholes')
    model_loc = settings.get_model_path('tables', 'boreholes')

    with open(model_loc, "wb") as file:
        pickle.dump(learner, file)


if __name__ == "__main__":
    create_dataset()
    #train(n_queries=2)