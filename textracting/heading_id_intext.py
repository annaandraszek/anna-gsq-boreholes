# headings dataset
# - exclude lines in toc
# - exclude lines in fig pages
# - exclude marginals

# inputs to trained model:
# lines in pages if pages not toc, not fig
# outputs:
# lines look like headings and all their info they came with
import re
import settings
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import pickle
import os
import eli5
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import textdistance
import spacy
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

def num2cyfra1(string):
    s = ''
    prev_c = ''
    i = 1
    for c in string:
        if re.match(r'[0-9]', c):
            if prev_c != 'num':
                s += ' cyfra' + str(i) + ' '
                i += 1
                prev_c = 'num'
        elif c == '.':
            s += ' punkt '
            prev_c = '.'
        else:
            s+= c
            #prev_c = 'char'
    return s


# technically an estimator.. but an estimator can't have acuracy
class Text2CNBPrediction(TransformerMixin, BaseEstimator):
    def fit(self, x, y):
        #x, y = check_X_y(x, y, accept_sparse=True)
        #self.n_features_ = x.shape[1]
        # add to pipeline: first step transforming all numbers to cyfra1 format
        text_clf = Pipeline([
            ('n2c', Num2Cyfra1()),
            ('tf', TfidfVectorizer(analyzer='word', ngram_range=(1,2))),
            ('cnb', ComplementNB(norm=True))])
        self.text_clf = text_clf.fit(x, y)
        self.feature_names_ = self.text_clf['tf'].get_feature_names()
        self.metrics(x, y)
        self.y_ = y

        return self

    def metrics(self, x, y):
        tf_words = self.text_clf['tf'].get_feature_names()

        pred = self.text_clf.predict(x)
        accuracy = accuracy_score(y, pred)
        print(confusion_matrix(y, pred))
        print('text2cnb accuracy: ', accuracy)
        print(classification_report(y, pred))
        right, wrong = 0, 0
        wrong_preds_x = []
        wrong_preds_y = []
        wrong_preds_pred = []
        for a, b, c in zip(y, pred, x):
            if a == b:
                right += 1
            else:
                wrong += 1
                wrong_preds_x.append(c)
                wrong_preds_y.append(a)
                wrong_preds_pred.append(b)

        wrong_dict = {'x': wrong_preds_x, 'y': wrong_preds_y, 'pred': wrong_preds_pred}
        wrong_df = pd.DataFrame(data=wrong_dict)
        return accuracy, wrong_df



    def transform(self, data):
        #check_is_fitted(self, 'n_features_')
        #data = check_array(data, accept_sparse=True)
        #if data.shape[1] != self.n_features_:
        #    raise ValueError('Shape of input is different from what was seen in `fit`')
        # self.data = data
        pred = self.text_clf.predict(data)
        return pd.DataFrame(pred)  # check what form this is in

    def get_feature_names(self):
        return self.feature_names_



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


#model = spacy.load('en_core_web_md')


# comparison of doc lines to toc headings
# comparing each line to each heading so a lot of computation
# returns highest similarities to a heading
# will have to add this method to heading_id_intext edit_dataset
def compare_lines2headings(lines, headings):
    if headings.shape[0] == 0:
        print('Headings are empty')
        return np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines))
    max_similarities = []
    for line in lines:
        ln_similarities = []
        ln_words = line.lower().split()
        for i, heading in headings.iterrows():  # save info whether the best comparison is to a heading or subheading
            hd_words = heading.Text.lower().split()
            # compare words
            similarity = textdistance.jaccard(ln_words, hd_words)  # intersection / union
            ln_similarities.append([similarity, heading.Heading, i])
        max = np.array(ln_similarities)[:, 0].argmax()
        bestsim = ln_similarities[max]
        if bestsim[0] == 0:  # if basically find no similarity
            bestsim = np.array([0, 0, 0])
        max_similarities.append(bestsim)
    max_similarities = np.array(max_similarities)
    return max_similarities[:, 0], max_similarities[:, 1], max_similarities[:,2]  # return similarity,type matched, and i of heading matched


def edit_dataset(dataset=settings.dataset_path + 'heading_id_intext_dataset.csv'):
    df = pd.read_csv(dataset)
    #df['WordCount'] = df.Text.apply(lambda x: len(x.split()))
    # need to reference heading_id_dataset.csv: DocId, LineText, Heading[0, 1, 2]
    toc_df = pd.read_csv(settings.dataset_path + 'processed_heading_id_dataset.csv')#, columns=['DocID', 'LineText', 'Heading'])
    toc_head_df = toc_df.loc[toc_df.Heading > 0]
    #df['MatchesHeading'], df['MatchesType'], df['MatchesI'] = pd.Series([]), pd.Series([]), pd.Series([])
    toc_head_df['Text'] = toc_head_df.apply(lambda x: str(x.SectionPrefix) + ' ' + x.SectionText, axis=1)
    series_mh = pd.Series()
    series_mt = pd.Series()
    series_mi = pd.Series()

    for docid in df.DocID.unique():
        doc_toc = toc_head_df.loc[toc_head_df.DocID == float(docid)]
        df_doc = df.loc[df.DocID == float(docid)]
        matches_heading, matches_type, matches_i = compare_lines2headings(df_doc.Text, doc_toc)
        print(len(matches_heading) == df_doc.shape[0], docid)
        series_mh = series_mh.append(pd.Series(matches_heading), ignore_index=True)
        series_mt = series_mt.append(pd.Series(matches_type), ignore_index=True)
        series_mi = series_mi.append(pd.Series(matches_i), ignore_index=True)

    df['MatchesHeading'], df['MatchesType'], df['MatchesI'] = series_mh, series_mt, series_mi

    # try:
    #     df['MatchesHeading'], df['MatchesType'], df['MatchesI'] = df.apply(
    #     lambda x: check_if_line_in_TOC(x.DocID, [x.Text], toc_head_df), axis=1)
    # except:
    #     print('oopsie')
    df.to_csv(dataset, index=False)


#def rm_empty(string):
#    return re.sub('^(|\s+)$', np.nan, str(string))


def data_prep(df, y=False):
#    df = df.apply(lambda x: rm_empty(x))
#    df = df.dropna()
    original_cols = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum', 'Text', 'Words2Width', 'WordsWidth', 'Width',
                     'Height', 'Left','Top', 'ContainsNum', 'Centrality', 'Heading', 'WordCount', 'MatchesHeading','MatchesType', 'MatchesI']

    df = pd.DataFrame(df, columns=original_cols)  # ordering as the fit, to not cause error in ColumnTranformer
    X = df.drop(columns=['DocID', 'LineNum', 'WordsWidth', 'NormedLineNum', 'Top', 'Heading', 'Centrality',  'MatchesI']) #'MatchesType',
    if y:
        Y = df.Heading
        return X, Y
    else:
        return X


def train(data=pd.read_csv(settings.dataset_path + 'heading_id_intext_dataset.csv'),
          model_file=settings.heading_id_intext_model_file):
    X, Y = data_prep(data, y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.20)

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
    y_true, y_pred = Y, clf.predict(X)
    report = classification_report(y_true, y_pred)
    print(report)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)

    #temp = pd.DataFrame(data=data)
    #temp['y_pred'] = y_pred
    #temp['correct'] = temp.Heading == temp.y_pred
    #cnb = Text2CNBPrediction().fit(X_train, y_train)
    #temp['text_transform'] = cnb.transform(temp['Text'])
    #print([(a, x, y) for a, x, y in zip(data.Text, y_true, y_pred) if x!=y])

    #cnb_transformer = clf['union'].transformers[0][1]
    #print(cnb_transformer.accuracy(return_wrong=True))

    #eli5.show_weights(clf)

    with open(model_file, "wb") as file:
        pickle.dump(clf, file)
    #cnb_predictor = Text2CNBPrediction()
    #Text2CNBPrediction.__module__ = "heading_id_intext"
    #with open("models/cnb_predictor.pkl", 'wb') as file:
    #    pickle.dump(cnb_predictor, file)
    #with open("models/num2cyfra1.pkl", 'wb') as file:
    #    pickle.dump(num2cyfra1, file)


def classify(data, model_file=settings.heading_id_intext_model_file):
    if not os.path.exists(model_file):
        train()
    # with open("models/cnb_predictor.pkl", 'rb') as file:
    #     cnb_predictor = pickle.load(file)
    # with open("models/num2cyfra1.pkl", 'rb') as file:
    #     num2cyfra1 = pickle.load(file)
    with open(model_file, "rb") as file:
        model = pickle.load(file)
    data = data_prep(data)
    pred = model.predict(data)
    return pred


def get_headings_intext(data):
    pred = classify(data)
    data['Heading'] = pred
    headings = data.loc[pred > 0]
    #return headings[['PageNum', 'LineNum', 'Text', 'Heading']]
    return headings.loc[headings.MatchesHeading > 0]


if __name__ == '__main__':
    data_path = settings.dataset_path + 'heading_id_intext_dataset.csv'
    #data = pd.read_csv(data_path)
    #create_dataset(data_path)
    #edit_dataset(data_path)
    data = pd.read_csv(data_path)
    train(data)
    preds = classify(data)
    x = 0
    for i, row in data.iterrows():
        if preds[i] != row.Heading:
            print(row.DocID, '\t', row.PageNum, ',', row.LineNum, '\t', row.Text, ' | ', row.Heading, ' | ', preds[i])
            x += 1
    print('Wrong classifications: ', x)