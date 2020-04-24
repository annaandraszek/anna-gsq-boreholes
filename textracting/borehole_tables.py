import pandas as pd
import settings
from io import StringIO
import glob
from textractor import texttransforming
import json
import pandas.errors
import os
import re
import numpy as np
from bookmarker import active_learning, machine_learning_helper as mlh
import pickle
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


name = "tables"
y_column = 'Class'
limit_cols = ['DocID', 'TableNum']


# extract tables from a csv
def get_tables(docid, bh=False, report_num=1):
    tablefile = settings.get_tables_file(docid, bh=bh, report_num=report_num)
    tablefile = tablefile.split('../')[1]
    if os.path.exists(tablefile):
        with open(tablefile, "r") as f:
            raw_tables = f.read()
    else:
        raise FileNotFoundError

    tables = []
    table = ""
    prev_ln_table = 0
    prev_ln_ln = 0
    lines = raw_tables.split('\n')
    for line in lines:
        if not line or line == '""':
            if len(table) > 0 and not prev_ln_table and not prev_ln_ln:
                tables.append(table)
                table = ""
            prev_ln_ln = 0
            prev_ln_table = 0
        else:
            prev_ln_ln = 1
            if 'Table: Table' in line:
                prev_ln_table = 1
            else:
                prev_ln_table = 0
                table += line+'\n'

    dfs = []
    for table in tables:
        data = StringIO(table)
        if not bh:
            try:
                df = pd.read_csv(data, sep='`')
            except pandas.errors.ParserError:
                continue
        else:
            df = pd.read_csv(data)
        df.dropna(axis=1, how="all", inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        #df = pd.DataFrame(table)
        #print(df.columns.values)
        dfs.append(df)
    return dfs


def create_dataset(docids=False, save=True):
    if docids:
        save = False
    if save:
        dataset = settings.get_dataset_path('tables', 'boreholes')
        dataset = dataset.split('../')[1]
    #docids = ['32730', '44448', '37802', '2646', '44603']
        docids = []
        lines_docs = glob.glob('training/fulljson/*.json')
        for lines_doc in lines_docs:
            docid = int(lines_doc.split('\\')[-1].replace('_1_fulljson.json', '').strip('cr_'))
            docids.append(docid)
    cols = ['DocID', 'TableNum', 'Content', 'FullTable']
    all_columns = pd.DataFrame(columns=cols)
    for id in docids:
        # try:
        #     texttransforming.save_tables_and_kvs(id)
        # except json.decoder.JSONDecodeError:
        #     print(id)
        #     continue
        tables = get_tables(id)
        #columns = pd.Series([table.columns.values for table in tables])
        full_tables = []
        for table in tables:
            t = table.to_numpy()
            t = t.astype(str)
            t = np.insert(t, 0, table.columns.values, 0)
            full_tables.append(t)

        tables_values = [list(table.columns.values) for table in tables]
        #exclude = ['Unnamed: ', 'nan']
        for t, i in zip(tables, range(len(tables))):
            for j, row in t.iterrows():
                tables_values[i] = np.concatenate((tables_values[i], row.values))
                tables_values[i] = [v for v in tables_values[i] if re.match(r'[A-z]+', str(v))]
                tables_values[i] = [v for v in tables_values[i] if 'Unnamed:' not in str(v)]
                tables_values[i] = [v for v in tables_values[i] if str(v) != 'nan']
        tables_values = pd.Series(tables_values)
        docids = pd.Series([id for x in range(len(tables_values))])
        tablenums = pd.Series([x + 1 for x in range(len(tables_values))])
        fulls = pd.Series(full_tables)
        series = [docids, tablenums, tables_values, fulls]
        iddf = pd.concat(series, axis=1)
        iddf.columns = cols
        #all_columns = all_columns.append(pd.Series(columns), ignore_index=True)
        all_columns = all_columns.append(iddf, ignore_index=True)
    if save:
        all_columns.to_csv(dataset, index=False)
        print('Done creating ', dataset)
    else:
        return all_columns


def list2str(lst):
    #table = ""
    #for e in lst:
    #    table += str(e) + ' '
    return lst[0]


def concat_tables(df):
    #print("original df\n", df)
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
    #series = series.apply(lambda x: list2str(x))


def train(n_queries=10, mode='boreholes'):
    datafile = settings.get_dataset_path(name, mode).split('../')[1]
    df = pd.read_csv(datafile)
    df = df.loc[df['Columns'] != '[]']

    clf = Pipeline([
        ('list2str', FunctionTransformer(concat_tables)),
        #('vect', CountVectorizer(ngram_range=(1, 2), min_df=0.01)),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=0.0025)),
        ('cnb', ComplementNB(alpha=0.2))
    ], verbose=True)
    accuracy, learner = active_learning.train(df, y_column, n_queries, clf, datafile, limit_cols=limit_cols,
                                              mode=mode)
    model_loc = settings.get_model_path(name, mode)

    with open(model_loc, "wb") as file:
        pickle.dump(learner, file)
    return learner


def get_borehole_tables(df, mode="boreholes", masked=False):
    if df['Content'].dtype == object:
        df['Content'] = df['Content'].astype(str)
    return mlh.get_classified(df, name, y_column, limit_cols, mode, masked)


class NoNaturalTablesError(Exception):
    pass


def get_bh_tables_from_docid(docids):
    if not isinstance(docids, list):  # in case only one docid is given
        docids = [docids]
    for id in docids:
        print("Getting borehole tables for ", id)
        df = create_dataset([id], save=False)
        df = df.loc[df['Content'].str.len() > 0]
        num_tables = df.shape[0]
        if num_tables == 0:
            raise NoNaturalTablesError('DocID has no natural tables')
        res = get_borehole_tables(df, masked=True)
        num_bh_tables = res.shape[0]
        print('Num of all tables: ', num_tables, ', num of borehole tables: ', num_bh_tables)
        tables = get_tables(id)
        bh_tables = []
        print("Borehole tables: ")
        for i in range(len(tables)):
            if i+1 in res['TableNum'].values:
                print(tables[i])
                bh_tables.append(tables[i])
                
    return bh_tables


def bh_tables_to_csv(docid):
    try:
        bh_tables = get_bh_tables_from_docid([docid])
    except NoNaturalTablesError as e:
        print(e)
        return
    file = settings.get_tables_file(docid, bh=True)
    file = file.strip(r'../')
    if os.path.exists(file):
        os.remove(file)  # because we will be using append, don't want a file to already exists
    i = 0
    for df in bh_tables:
        i += 1
        with open(file, 'a') as f:
            startval = 'Table: Table ' + str(i)
            startseries = pd.Series([startval])
            startseries.to_csv(f, index=False, header=False)
            df.to_csv(f, index=False)
            blnk_ln = pd.Series('')
            blnk_ln.to_csv(f, index=False, header=False)
    print('Saved ', docid, ' bh tables to file')
    return


def save_all_bh_tables():
    docids = []
    lines_docs = glob.glob('training/tables/*.csv')
    for lines_doc in lines_docs:
        docid = int(lines_doc.split('\\')[-1].replace('_1_tables.csv', '').strip('cr_'))
        docids.append(docid)
    for id in docids:
        bh_tables_to_csv(id)
    print('Saved all bh tables')


# def create_bh_dataset():
#     dataset = settings.get_dataset_path('tables', 'boreholes')
#     dataset = dataset.split('../')[1]
#     docids = []
#     lines_docs = glob.glob('training/tables_bh/*.csv')
#     for lines_doc in lines_docs:
#         docid = int(lines_doc.split('\\')[-1].replace('_1_tables_bh.csv', '').strip('cr_'))
#         docids.append(docid)
#
#     for id in docids:
#         tables = get_tables(id)
#         for table in tables:
#             num_rows, num_cols = table.shape()
#             rowcol_ratio = num_rows // num_cols
#             row_similarities = []
#             col_similarities = []
#             for i, j in table.iterrows():
#                 # get row similarity in char textdistance, and average difference in char length differences


def table_similarity(docid):
    bhtables = get_tables(docid, bh=True)
    for table in bhtables:
        rows, cols = table.shape
        t = table.to_numpy()
        t = np.insert(t, 0, table.columns.values)#, 0)
        t = t.astype(str)
        #t = t.tolist()
        vectorizer = CountVectorizer(analyzer='char')
        X = vectorizer.fit_transform(t)
        sim = cosine_similarity(X)
        rowsims = []
        colsims = []
        for r in range(rows+1):
            lbound = r*cols
            ubound = cols*(r+1)
            row = t[lbound:ubound]
            print(lbound, ubound)
            rowvec = vectorizer.transform(row)
            rowsim = cosine_similarity(rowvec)
            #print(rowsim)
            rowsims.append(rowsim)
        for c in range(cols):
            cis = [r*cols + c for r in range(rows+1)]
            col = [t[ci] for ci in cis]
            print(cis)
            colvec = vectorizer.transform(col)
            colsim = cosine_similarity(colvec)
            #print(rowsim)
            colsims.append(colsim)
        print('all rows')

#def extract_bh(df):
    # extract bh references from a table



if __name__ == "__main__":
    #create_dataset()

    # dataset = settings.get_dataset_path('tables', 'boreholes')
    # dataset = dataset.split('../')[1]
    # df = pd.read_csv(dataset)
    # prev_dataset = 'datasets/boreholes/tables_dataset_ann.csv'
    # df = mlh.add_legacy_y(prev_dataset=prev_dataset, df=df, y_column='Class', page=False, table=True)
    # df.to_csv(dataset, index=False)

    #train(n_queries=1)
    #active_learning.automatically_tag('tables', get_borehole_tables, 'Class', mode='boreholes_production')

    #train(n_queries=0)

    # df = create_dataset(['44448'], save=False)
    # df = df.loc[df['Columns'].str.len() > 0]
    # res = get_borehole_tables(df, masked=True)
    # print(res)

    #get_bh_tables_from_docid(['2646', '44448', '32730', '37802', '44603'])
    #bh_tables_to_csv('35454')
    save_all_bh_tables()
    #bhtables = get_tables('35454', bh=True)
    #table_similarity('35454')
