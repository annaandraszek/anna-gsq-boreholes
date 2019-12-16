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


def create_dataset():
    df = pd.DataFrame(columns=['DocID', 'PageNum', 'MedConfidence', 'MedLineLen', 'ContainsFigWord', 'ContainsFigLn', 'FigPos', 'FigPage', 'Comments'])
    pageinfos = sorted(glob.glob('training/restructpagelineinfo/*'))
    pagelines = sorted(glob.glob('training/restructpagelines/*'))

    for pagesinfo, pageslines in zip(pageinfos, pagelines):
        pi = json.load(open(pagesinfo))
        pl = json.load(open(pageslines))
        docset = np.zeros((len(pi.items()), 8))
        docid = pagesinfo.split('\\')[-1].replace('_1_pageinfo.json', '')

        for info, lines, j, in zip(pi.items(), pl.items(), range(len(pi.items()))):
            fig = 0
            figln = 0
            figlnpos = -1
            confs = []
            linelens = []
            for line in lines[1]:
                confs.append(info[1]['Confidence'])
                linelens.append(len(line))
                if 'figure' in line.lower():
                    fig = 1
                    if re.search(r'Figure\s\d+\.*\s\w', line):
                        figln = 1
                        figlnpos = j / len(lines[1])
            medconf = np.median(np.array(confs))
            medlineln = np.median(np.array(linelens))

            docset[j] = np.array([docid.strip('cr_'), info[1]['Page'], medconf, medlineln, fig, figln, figlnpos, 0])
        pgdf = pd.DataFrame(data=docset, columns=['DocID', 'PageNum', 'MedConfidence', 'MedLineLen', 'ContainsFigWord', 'ContainsFigLn', 'FigPos', 'FigPage'])
        df = df.append(pgdf, ignore_index=True)
    return df


def data_prep(data, y=False):
    data = data.drop(['Comments'], axis=1)
    data = data.dropna()
    X = data.drop(['DocID', 'FigPage'], axis=1)
    if y:
        Y = data.TOCPage
        return X, Y
    return X


def train(data):
    X, Y = data_prep(data, y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(accuracy)
    #tree.plot_tree(clf, feature_names=['PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord'], class_names=True, filled=True)
    #plt.show()
    with open(settings.fig_tree_model_file, "wb") as file:
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
    dataset = create_dataset()
    dataset.to_csv('fig_dataset.csv', index=False)
    print(dataset)
    #train(dataset)
    #fig_pages = get_fig_pages()
    #print(fig_pages)
