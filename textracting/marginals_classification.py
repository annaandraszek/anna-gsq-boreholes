import numpy as np
import pandas as pd
import json
import glob
import settings
import re
from sklearn import tree, naive_bayes, ensemble
import sklearn
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
import seaborn as sns
import graphviz


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


def create_dataset():
    columns = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum','Text', 'Words2Width', 'WordsWidth', 'Width', 'Height', 'Left', 'Top', 'ContainsNum',
               'ContainsTab', 'ContainsPage', 'Centrality']
    pageinfos = glob.glob('training/restructpageinfo/*')
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
                               contains_page, centrality])

            temp = pd.DataFrame(data=docset, columns=columns)
            temp['NormedLineNum'] = (temp['LineNum'] - min(temp['LineNum'])) / (max(temp['LineNum']) - min(temp['LineNum']))
            df = df.append(temp, ignore_index=True)

    unnormed = np.array(df['Centrality'])
    normalized = (unnormed - min(unnormed)) / (max(unnormed) - min(unnormed))
    df['Centrality'] = normalized
    df['Marginal'] = 0
    return df


def create_individual_dataset(docid):
    columns = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum','Text', 'Words2Width', 'WordsWidth', 'Width', 'Height', 'Left', 'Top', 'ContainsNum',
               'ContainsTab', 'ContainsPage', 'Centrality', 'Marginal']
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


def data_prep(data, y=False):
    data.dropna(inplace=True)
    data.drop(data[data['Width'] < 0].index, inplace=True)
    X = data.drop(['DocID', 'LineNum', 'Text', 'Marginal'], axis=1)  # PageNum?
    #X = X.drop(['Words2Width'], axis=1)  # temporarily
    if y:
        Y = data['Marginal']
        return X, Y
    return X


def train(data=pd.read_csv(settings.get_dataset_path('marginal_lines'))): #, model='forest'):
    X, Y = data_prep(data, y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33)
    #if 'forest' in model:
    clf = ensemble.RandomForestClassifier(n_estimators=12)
    model_file = settings.get_model_path('marginal_lines') #settings.marginals_model_file_forest
    # elif 'CNB' in model:
    #     clf = naive_bayes.ComplementNB()
    #     model_file = settings.marginals_model_file_CNB
    # else:
    #     clf = tree.DecisionTreeClassifier()
    #     model_file = settings.marginals_model_file_tree
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(accuracy)
    #tree.plot_tree(clf, feature_names=['PageNum', 'NormedLineNum', 'WordsWidth', 'Width', 'Height', 'Left', 'Top', 'ContainsNum',
    #           'ContainsTab', 'ContainsPage', 'Centrality', ], class_names=True, filled=True)
    #plt.show()
    #plt.savefig(settings.result_path + 'marginals_tree.png')

    # if 'tree' in model:
    #     dot_data = tree.export_graphviz(clf,  feature_names=['PageNum', 'NormedLineNum', 'Words2Width', 'WordsWidth', 'Width', 'Height',
    #                                                          'Left', 'Top', 'ContainsNum', 'ContainsTab', 'ContainsPage',
    #                                                          'Centrality', ], class_names=True, filled=True,
    #                                     max_depth=4) # out_file=settings.result_path + 'marginals_tree.png',
    #     graph = graphviz.Source(dot_data)
    #     graph.render("marginals")#.jpeg")


    #cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    cm = sklearn.metrics.plot_confusion_matrix(clf, X_test, y_test)
    print(cm.confusion_matrix)
    #df_cm = pd.DataFrame(cm)
    #sns.heatmap(df_cm, annot=True)
    #plt.show()
    with open(model_file, "wb") as file:
        pickle.dump(clf, file)

    # display the wrong predictions
    y_test = y_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    #print(y_pred.size, y_test.size, X_test.size)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    for i in range(y_pred.size):
        if y_pred[i] != y_test[i]:
            print('Predicted: ', y_pred[i], 'Actual: ', y_test[i], '\nX: ', X_test.iloc[[i]])


def classify(data):
    if not os.path.exists(settings.marginals_model_file_forest):
        train(data)
    with open(settings.marginals_model_file_forest, "rb") as file:
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
    file = 'marginals_dataset_v2.csv'
    data = pd.read_csv(settings.dataset_path + file)
    get_marginals(data)
    #train(data, model='forest')

    #classes = classify(data)
    #data['Marginal'] = classes
    #data.to_csv(settings.dataset_path + 'marginals_dataset_v2_tagged.csv', index=False)
