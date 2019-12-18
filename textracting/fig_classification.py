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

def create_dataset():
    df = pd.DataFrame(columns=['DocID', 'PageNum', 'MedConfidence', 'AvgConfidence', 'RangeConfidence', 'IQRConfidence','MedLineLen', 'ContainsFigWord', 'ContainsFigLn', 'FigPos', 'FigPage'])
    pageinfos = sorted(glob.glob('training/restructpageinfo/*'))
    pagelines = sorted(glob.glob('training/restructpagelines/*'))

    for pagesinfo, pageslines in zip(pageinfos, pagelines):
        #pagesinfo = 'training\\restructpageinfo\\cr_26114_1_restructpageinfo.json'
        #pageslines = 'training\\restructpagelines\\cr_26114_1_restructpagelines.json'
        pi = json.load(open(pagesinfo))
        pl = json.load(open(pageslines))
        docset = np.zeros((len(pi.items()), 11))
        docid = pagesinfo.split('\\')[-1].replace('_1_restructpageinfo.json', '')

        for info, lines, i in zip(pi.items(), pl.items(), range(len(pi))):
            fig = 0
            figln = 0
            figlnpos = -1
            confs = []
            linelens = []
            for line, inf, j in zip(lines[1], info[1], range(len(lines[1]))):
                confs.append(inf['Confidence'])
                linelens.append(len(line))
                if 'figure' in line.lower():
                    fig = 1
                    if figlnpos == -1:
                        figlnpos = j / len(lines[1])
                    if re.search(r'Figure\s\d+\.*\s*\w*', line) or re.search(r'FIGURE\s\d+\.*\s*\w*', line):
                        figln = 1
                        figlnpos = j / len(lines[1])
            medconf = np.median(np.array(confs))
            avgconf = np.average(np.array(confs))
            rangeconf = np.max(np.array(confs)) - np.min(np.array(confs))
            medlineln = np.median(np.array(linelens))

            q75, q25 = np.percentile(np.array(confs), [75, 25])
            iqr = q75 - q25

            docset[i] = np.array([docid.strip('cr_'), info[0], medconf, avgconf, rangeconf, iqr, medlineln, fig, figln, figlnpos, 0])
        pgdf = pd.DataFrame(data=docset, columns=['DocID', 'PageNum', 'MedConfidence', 'AvgConfidence', 'RangeConfidence', 'IQRConfidence','MedLineLen', 'ContainsFigWord', 'ContainsFigLn', 'FigPos', 'FigPage'])
        df = df.append(pgdf, ignore_index=True)
    return df


def data_prep(data, y=False, limited_cols=None):
    X = data.drop(['DocID', 'FigPage'], axis=1)

    if limited_cols:
        X = X.drop(limited_cols, axis=1)

    if y:
        Y = data['FigPage']
        return X, Y
    return X


def train(data, model_file=settings.fig_tree_model_file, limited_cols=None):
    X, Y = data_prep(data, y=True, limited_cols=limited_cols)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(accuracy)
    #tree.plot_tree(clf, feature_names=['PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord'], class_names=True, filled=True)
    #plt.show()
    with open(model_file, "wb") as file:
        pickle.dump(clf, file)


def train_several(X_train, X_test, y_train, y_test, limited_col=None):
    if limited_col:
        col = limited_col
        xtra = X_train.drop([col], axis=1)
        xtes = X_test.drop([col], axis=1)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(xtra, y_train)
        y_pred = clf.predict(xtes)
        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        print(accuracy)
        tree.plot_tree(clf, feature_names=xtra.columns, class_names=['Not figure', 'Figure'], filled=True)
        plt.savefig('fig_tree_' + limited_col + '.png')
        with open("-" + col + '_' + settings.fig_tree_model_file, "wb") as file:
            pickle.dump(clf, file)
    else:
        # control
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        print(accuracy)
        tree.plot_tree(clf, feature_names=X_train.columns, class_names=['Not figure', 'Figure'], filled=True)
        plt.savefig('fig_tree_1.png')
        model_file = settings.fig_tree_model_file
        with open(model_file, "wb") as file:
            pickle.dump(clf, file)


def classify_page(data):
    if not os.path.exists(settings.fig_tree_model_file):
        train(data)
    with open(settings.fig_tree_model_file, "rb") as file:
        model = pickle.load(file)
    data = data_prep(data)
    pred = model.predict(data)
    return pred


def get_fig_pages(data_file='fig_dataset.csv'):
    df = pd.read_csv(data_file)
    classes = classify_page(df)
    mask = np.array([True if i==1 else False for i in classes])
    fig_pages = df[mask]
    return fig_pages


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

    train_several(X_train, X_test, y_train, y_test)
    train_several(X_train, X_test, y_train, y_test, limited_col='MedConfidence')
    train_several(X_train, X_test, y_train, y_test, limited_col='AvgConfidence')

    #fig_pages = get_fig_pages()
    #print(fig_pages)
