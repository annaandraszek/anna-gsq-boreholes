import numpy as np
import pandas as pd
import json
import glob
import settings
import re
from sklearn import tree
import sklearn
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
import seaborn as sns

def create_dataset():
    columns = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum','Text', 'WordsWidth', 'Width', 'Height', 'Left', 'Top', 'ContainsNum',
               'ContainsTab', 'ContainsPage', 'Centrality', 'Marginal']
    pageinfos = glob.glob('training/restructpageinfo/*')
    df = pd.DataFrame(columns=columns)
    for pagesinfo in pageinfos:
        pi = json.load(open(pagesinfo))
        docid = pagesinfo.split('\\')[-1].replace('_1_restructpageinfo.json', '').strip('cr_')
        for info in pi.items():
            docset = []
            page = info[0]
            lines = len(info[1])
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

                docset.append([docid, int(page), line['LineNum'], 0, line['Text'], line['WordsWidth'],
                               bb['Width'], bb['Height'], bb['Left'], bb['Top'], contains_num, contains_tab,
                               contains_page, centrality, 0])

            temp = pd.DataFrame(data=docset, columns=columns)
            temp['NormedLineNum'] = (temp['LineNum'] - min(temp['LineNum'])) / (max(temp['LineNum']) - min(temp['LineNum']))
            df = df.append(temp, ignore_index=True)

    unnormed = np.array(df['Centrality'])
    normalized = (unnormed - min(unnormed)) / (max(unnormed) - min(unnormed))
    df['Centrality'] = normalized
    return df


def data_prep(data, y=False):
    data.dropna(inplace=True)
    X = data.drop(['DocID', 'LineNum', 'Text', 'Marginal'], axis=1)  # PageNum?
    if y:
        Y = data['Marginal']
        return X, Y
    return X


def train(data, model_file=settings.marginals_model_file):
    X, Y = data_prep(data, y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(accuracy)
    #tree.plot_tree(clf, feature_names=['PageNum', 'NormedLineNum', 'WordsWidth', 'Width', 'Height', 'Left', 'Top', 'ContainsNum',
    #           'ContainsTab', 'ContainsPage', 'Centrality', ], class_names=True, filled=True)
    #plt.show()
    #plt.savefig(settings.result_path + 'marginals_tree.png')

    #cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    cm = sklearn.metrics.plot_confusion_matrix(clf, X_test, y_test)
    print(cm.confusion_matrix)
    #df_cm = pd.DataFrame(cm)
    #sns.heatmap(df_cm, annot=True)
    #plt.show()
    with open(model_file, "wb") as file:
        pickle.dump(clf, file)


def classify_line(data):
    if not os.path.exists(settings.marginals_model_file):
        train(data)
    with open(settings.marginals_model_file, "rb") as file:
        model = pickle.load(file)
    data = data_prep(data)
    pred = model.predict(data)
    return pred


if __name__ == "__main__":
    #df = create_dataset()
    #df.to_csv(settings.dataset_path + 'marginals_dataset.csv', index=False)

    matplotlib.rcParams['figure.figsize'] = (20, 20)
    file = 'marginals_dataset.xlsx'
    data = pd.read_excel(settings.dataset_path + file)
    train(data)
