# headings dataset
# - exclude lines in toc
# - exclude lines in fig pages
# - exclude marginals

# inputs to trained model:
# lines in pages if pages not toc, not fig
# outputs:
# lines look like headings and all their info they came with
import pandas as pd
import re
import settings
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import pickle
import os
import eli5
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from lstm_heading_identification import num2cyfra1
import pandas as pd


class Text2CNBPrediction(TransformerMixin, BaseEstimator):
    def fit(self, x, y):
        # add to pipeline: first step transforming all numbers to cyfra1 format
        text_clf = Pipeline([
            ('n2c', Num2Cyfra1()),
            ('tf', TfidfVectorizer()),
            ('cnb', ComplementNB(norm=True))])
        self.text_clf = text_clf.fit(x, y)
     #   self.feature_names = self.text_clf['tf'].get_feature_names()
        # self.y = y
        return self

    def transform(self, data):
        pred = self.text_clf.predict(data)
        # self.data = data
        return pd.DataFrame(pred)  # check what form this is in

    #def get_feature_names(self):
    #    return self.feature_names

    # def accuracy(self, return_wrong=False):
    #     pred = self.text_clf.predict(self.data)
    #     right, wrong = 0, 0
    #
    #     if return_wrong:
    #         wrong_preds_x = []
    #         wrong_preds_y = []
    #         wrong_preds_pred = []
    #     for a, b, c in zip(self.y, pred, self.data):
    #         if a == b:
    #             right += 1
    #         else:
    #             wrong += 1
    #             if return_wrong:
    #                 wrong_preds_x.append(c)
    #                 wrong_preds_y.append(a)
    #                 wrong_preds_pred.append(b)
    #
    #     accuracy = right/(right+wrong)
    #     if return_wrong:
    #         wrong_dict = {'x': wrong_preds_x, 'y': wrong_preds_y, 'pred': wrong_preds_pred}
    #         wrong_df = pd.DataFrame(data=wrong_dict)
    #         return accuracy, wrong_df
    #     return accuracy


class Num2Cyfra1(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        data = data.apply(lambda x: num2cyfra1(x))
        return data  # check what form this is in



def contains_num(x):
    if re.search('^[0-9]+.*?\s\w', str(x)):
        return 1
    else:
        return 0


# kinda awkward because you're creating it from another dataset...so you're depending on that one to be complete
def create_dataset(datafile=settings.dataset_path + 'heading_id_intext_dataset.csv', docid=False):
    sourcefile = settings.dataset_path + 'marginals_dataset_v2.csv'
    df = pd.read_csv(sourcefile)

    if docid:
        df = df.loc[df['DocID'] == float(docid)]
    # remove ContainsTab, ContainsPage
    df = df.drop(['ContainsTab', 'ContainsPage'], axis=1)
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
    df = df.drop(index=to_drop)

    # update contains num to just re.search('[0-9]+')
    df['ContainsNum'] = df.Text.apply(lambda x: contains_num(x))
    # add column: line word count
    df['WordCount'] = df.Text.apply(lambda x: len(x.split()))

    #print(new_df)
    if not docid:
        df.to_csv(datafile, index=False)
    df['Heading'] = 0
    return df
    # manually annotate, or, send to classifier


def edit_dataset(dataset=settings.dataset_path + 'heading_id_intext_dataset.csv'):
    df = pd.read_csv(dataset)
    df['WordCount'] = df.Text.apply(lambda x: len(x.split()))
    df.to_csv(dataset, index=False)


#def rm_empty(string):
#    return re.sub('^(|\s+)$', np.nan, str(string))


def data_prep(df, y=False):
#    df = df.apply(lambda x: rm_empty(x))
#    df = df.dropna()
    X = df.drop(columns=['DocID', 'Top', 'Heading'])
    if y:
        Y = df.Heading
        return X, Y
    else:
        return X



def train(data=pd.read_csv(settings.dataset_path + 'heading_id_intext_dataset.csv'),
          model_file=settings.heading_id_intext_model_file):
    X, Y = data_prep(data, y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.25)

    clf = Pipeline([
        ('text', ColumnTransformer([
            ('cnb', Text2CNBPrediction(), 'Text')
        ], remainder="passthrough")),
        ('forest', RandomForestClassifier())
    ], verbose=True)

    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    report = classification_report(Y, clf.predict(X))
    print(report)

    #cnb_transformer = clf['union'].transformers[0][1]
    #print(cnb_transformer.accuracy(return_wrong=True))

    #eli5.show_weights(clf)

    with open(model_file, "wb") as file:
        pickle.dump(clf, file)
    cnb_predictor = Text2CNBPrediction()
    #Text2CNBPrediction.__module__ = "model_maker"
    with open("cnb_predictor.pkl", 'wb') as file:
        pickle.dump(cnb_predictor, file)


def classify(data, model_file=settings.heading_id_intext_model_file):
    if not os.path.exists(model_file):
        train()
    with open(model_file, "rb") as file:
        model = pickle.load(file)
    with open("cnb_predictor.pkl", 'rb') as file:
        cnb_predictor = pickle.load(file)
    data = data_prep(data)
    pred = model.predict(data)
    return pred


def get_headings_intext(docid, dataset=None):
    try:
        if not dataset:
            data = create_dataset(docid=docid)
    except ValueError:
        data = dataset
    pred = classify(data)
    data['Heading'] = pred
    headings = data.loc[pred > 0]
    return headings[['PageNum', 'LineNum', 'Text', 'Heading']]


if __name__ == '__main__':
    data_path = settings.dataset_path + 'heading_id_intext_dataset.csv'
    #data = pd.read_csv(data_path)
    #create_dataset(data_path)
    edit_dataset(data_path)
    data = pd.read_csv(data_path)
    #train(data)
    preds = classify(data)
    x = 0
    for i, row in data.iterrows():
        if preds[i] != row.Heading:
            print(row.DocID, '\t', row.PageNum, ',', row.LineNum, '\t', row.Text, ' | ', row.Heading, ' | ', preds[i])
            x += 1
    print('Wrong classifications: ', x)