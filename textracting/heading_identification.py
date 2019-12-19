from toc_classification import get_toc_pages
import glob
import pandas as pd
import json
import numpy as np
import re
import sklearn
import pickle
import settings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def split_prefix(string):
    s = re.split(r'(^[0-9]+\.*[0-9]*\.*[0-9]*)', string, 1)
    if len(s) == 1:
        s = ['', s[0]]
    elif len(s) == 3:
        s = [s[-2], s[-1]]
    return s


def split_pagenum(string):
    s = re.split(r'(\t[0-9]+$)', string, 1) # if $ doesn't work try \Z
    if len(s) == 1:
        s = [s[0], '']
    elif len(s) == 3:
        s = [s[0], s[1]]
    return s


def num2cyfra1(string):
    s = ''
    prev_c = ''
    i = 1
    for c in string:
        if re.match(r'[0-9]', c):
            if prev_c != 'num':
                s += 'cyfra' + str(i) + ' '
                i += 1
                prev_c = 'num'
        elif c == '.':
            s += 'punkt '
            prev_c = '.'
    return s


def num2cyfra(string):
    s = ''
    prev_c = ''
    for c in string:
        if re.match(r'[0-9]', c):
            if prev_c != 'num':
                s += 'cyfra '
                prev_c = 'num'
        elif c == '.':
            s += 'punkt '
            prev_c = '.'
    return s

def pre_process_id_dataset(pre='cyfra'):
    df = pd.read_csv("heading_id_dataset.csv")
    # break up the LineText column into SectionPrefix, SectionText, and SectionPage
    newdf = pd.DataFrame(columns=['DocID', 'LineNum', 'SectionPrefix', 'SectionText', 'SectionPage', 'Heading'])
    newdf.DocID = df.DocID
    newdf.LineNum = df.LineNum
    newdf.Heading = df.Heading

    newdf.SectionPrefix, newdf.SectionText = zip(*df.LineText.map(split_prefix))
    newdf.SectionText, newdf.SectionPage = zip(*newdf.SectionText.map(split_pagenum))

    if pre == 'cyfra1':
        newdf.SectionPrefix = newdf.SectionPrefix.apply(lambda x: num2cyfra1(x))
        newdf.SectionPage = newdf.SectionPage.apply(lambda x: num2cyfra1(x))
    else:
        newdf.SectionPrefix = newdf.SectionPrefix.apply(lambda x: num2cyfra(x))
        newdf.SectionPage = newdf.SectionPage.apply(lambda x: num2cyfra(x))

    newdf.replace('', np.nan, inplace=True)
    newdf.dropna(inplace=True, subset=['SectionText'])
    newdf.replace(np.nan, '', inplace=True)  # nan values cause issues when adding columns

    newdf.SectionText = newdf.SectionPrefix + newdf.SectionText + newdf.SectionPage
    newdf.drop(axis=1, columns=['SectionPrefix', 'SectionPage'], inplace=True)
    return newdf


def create_identification_dataset():
    df = pd.DataFrame(columns=['DocID', 'LineNum', 'LineText', 'Heading'])
    lines_docs = sorted(glob.glob('training/restructpagelines/*'))
    toc_pages = get_toc_pages()
    for lines_doc in lines_docs:
        pages = json.load(open(lines_doc))
        docid = int(lines_doc.split('\\')[-1].replace('_1_restructpagelines.json', '').strip('cr_'))
        tocpg = toc_pages.loc[toc_pages['DocID'] == docid]
        try:
            page = tocpg.PageNum.values[0]
            for lines in pages.items():
                if lines[0] == str(page):
                    docset = []
                    for line, i in zip(lines[1], range(len(lines[1]))):
                        heading = 0
                        if re.match(r'^([0-9]+\.[0-9]+\s+\w+)', line):
                            heading = 2
                        elif re.match(r'^[0-9]+\.*\s+\w+', line):
                            heading = 1

                        docset.append([docid, i, line, heading])
                    pgdf = pd.DataFrame(data=docset, columns=['DocID', 'LineNum', 'LineText', 'Heading'])
                    df = df.append(pgdf, ignore_index=True)
        except IndexError:
            print("IndexError ", tocpg, docid)
    df.to_csv("heading_id_dataset.csv", index=False)
    return df


def data_prep(df, y=False):
    X = df.SectionText
    if y:
        Y = df.Heading
        return X, Y
    else:
        return X


def train(data, pre='cyfra', model_file=settings.headid_nb_model_file):
    X, Y = data_prep(data, y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33)
    clf = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', ComplementNB(norm=True))])  # pipeline of fit/transforms

    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(accuracy)
    report = classification_report(y_train, clf.predict(X_train))
    print(report)
    with open(pre + '_CNBreport.txt', "w") as r:
        r.write(report)
    with open(pre + model_file, "wb") as file:
        pickle.dump(clf, file)


def predict(inputs):
    if isinstance(inputs, str):
        inputs = [inputs]
    with open(settings.headid_nb_model_file, "rb") as file:
        model = pickle.load(file)
    pred = model.predict(inputs)
    return pred


if __name__ == "__main__":
    #df = create_identification_dataset()
    pre = 'cyfra'
    df = pre_process_id_dataset(pre)
    df.to_csv('processed_heading_id_dataset_' + pre + '.csv', index=False)
    #df = pd.read_csv('processed_heading_id_dataset_cyfra1.csv')
    train(df, pre)
    #to_predict = df.SectionText
    #p = predict(to_predict)
    #for i, j in zip(to_predict, p):
    #    print(i, j)
