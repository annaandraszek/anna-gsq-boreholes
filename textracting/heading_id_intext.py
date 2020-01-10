

# headings dataset
# - exclude lines in toc
# - exclude lines in fig pages
# - exclude marginals


# inputs to trained model:
# lines in pages if pages not toc, not fig
# outputs:
# lines look like headings and all their info they came with
import numpy as np
import pandas as pd
import re
import settings
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

def contains_num(x):
    if re.search('^[0-9]+.*?\s\w', str(x)):
        return 1
    else:
        return 0

def create_dataset(datafile=settings.dataset_path + 'heading_id_intext_dataset.csv'):
    sourcefile = settings.dataset_path + 'marginals_dataset_v2.csv'
    df = pd.read_csv(sourcefile)
    # remove ContainsTab, ContainsPage
    df = df.drop(['ContainsTab', 'ContainsPage'], axis=1)
    # update contains num to just re.search('[0-9]+')
    df.ContainsNum = df.Text.apply(lambda x: contains_num(x))
    # remove rows with Marginal == 1 or 2. then remove marginal column
    df = df.loc[df.Marginal == 0]
    df = df.drop(['Marginal'], axis=1)
    # find ALL the toc pages and remove their lines from the dataset
    # find ALL the fig pages and remove their lines from the dataset
    toc_dataset = pd.read_csv(settings.dataset_path + 'toc_dataset.csv')
    fig_dataset = pd.read_csv(settings.dataset_path + 'fig_dataset.csv')
    tocs = toc_dataset.loc[toc_dataset.TOCPage == 1]
    figs = fig_dataset.loc[fig_dataset.FigPage == 1]
    toc_tuples = [(id, page) for id, page in zip(tocs.DocID, tocs.PageNum)]
    fig_tuples = [(id, page) for id, page in zip(figs.DocID, figs.PageNum)]
    to_drop = []
    for i, row in df.iterrows():
        if (row.DocID, row.PageNum) in toc_tuples or (row.DocID, row.PageNum) in fig_tuples:
            to_drop.append(i)

    # add column: line word count
    df['WordCount'] = df.Text.apply(lambda x: len(x.split()))

    new_df = df.drop(index=to_drop)
    print(new_df)
    new_df.to_csv(datafile, index=False)

    # manually annotate


def edit_dataset(dataset=settings.dataset_path + 'heading_id_intext_dataset.csv'):
    df = pd.read_csv(dataset)
    df['WordCount'] = df.Text.apply(lambda x: len(x.split()))
    df.to_csv(dataset, index=False)


def data_prep(df, y=False):
    X = df.drop(columns=['DocID', 'Top', 'Heading'])
    if y:
        Y = df.Heading
        return X, Y
    else:
        return X


class Text2CNBPrediction(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        text_clf = Pipeline([('tf', TfidfVectorizer()), ('cnb', ComplementNB())])
        self.text_clf = text_clf.fit(x, y)
        return self

    def transform(self, data):
        pred = self.text_clf.predict(data)
        return pd.DataFrame(pred)  # check what form this is in


def find_bad(x):
    try:
        float(x)
    except ValueError:
        return "I'M BADDDDDDDDDDDDDDDDDD"

def train(data, model_file=settings.heading_id_intext_model_file):
    # one main pipeline, made up of a columntranformer containing little pipelines for different operations on data (text vs other)

    X, Y = data_prep(data, y=True)
    #X.Centrality = X.Centrality.apply(lambda x: find_bad(x))
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33)

    clf = Pipeline([
        ('union', ColumnTransformer([
            #('clf', Pipeline([
            #    ('tf', TfidfVectorizer()),
            #    ('cnb', ComplementNB())
            ('text', Text2CNBPrediction()
            #])
            , 'Text')
        ], remainder="passthrough")),
        ('forest', RandomForestClassifier())
    ], verbose=True)

    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    report = classification_report(y_train, clf.predict(X_train))
    print(report)
    #with open(pre + '_CNBreport.txt', "w") as r:
    #    r.write(report)
    #with open(pre + model_file, "wb") as file:
    #    pickle.dump(clf, file)


if __name__ == '__main__':
    data_path = settings.dataset_path + 'heading_id_intext_dataset.csv'
    data = pd.read_csv(data_path)
    #create_dataset(data_path)
    #edit_dataset(data_path)
    train(data)