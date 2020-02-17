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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import pickle
import os
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import textdistance
import numpy as np
import active_learning
import machine_learning_helper as mlh

name = 'heading_id_intext'
y_column = 'Heading'
limit_cols = ['DocID', 'LineNum', 'WordsWidth', 'NormedLineNum', 'Top', 'Heading', 'Centrality', 'MatchesI']


def num2cyfra1(string):
    s = ''
    prev_c = ''
    i = 1
    for c in string:
        if re.match(r'[0-9]', c):
            if prev_c != 'num':
                if c != '0':  # to stop eg. 1.0 being tagged as cyfra1 punkt cyfra2 like a subheading
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
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        text_clf = Pipeline([
            ('n2c', Num2Cyfra1()),
            ('tf', TfidfVectorizer(analyzer='word', ngram_range=(1,2))),
            ('cnb', ComplementNB(norm=True))])
        self.text_clf = text_clf.fit(x, y)
        self.feature_names_ = self.text_clf['tf'].get_feature_names()
        # self.metrics(x, y)
        self.y_ = y

        return self

    # def metrics(self, x, y):
    #     tf_words = self.text_clf['tf'].get_feature_names()
    #
    #     pred = self.text_clf.predict(x)
    #     accuracy = accuracy_score(y, pred)
    #     print(confusion_matrix(y, pred))
    #     print('text2cnb accuracy: ', accuracy)
    #     print(classification_report(y, pred))
    #     right, wrong = 0, 0
    #     wrong_preds_x = []
    #     wrong_preds_y = []
    #     wrong_preds_pred = []
    #     for a, b, c in zip(y, pred, x):
    #         if a == b:
    #             right += 1
    #         else:
    #             wrong += 1
    #             wrong_preds_x.append(c)
    #             wrong_preds_y.append(a)
    #             wrong_preds_pred.append(b)
    #
    #     wrong_dict = {'x': wrong_preds_x, 'y': wrong_preds_y, 'pred': wrong_preds_pred}
    #     wrong_df = pd.DataFrame(data=wrong_dict)
    #     return accuracy, wrong_df



    def transform(self, data):
        pred = self.text_clf.predict(data)
        return pd.DataFrame(pred)  # check what form this is in

    def get_feature_names(self):
        return self.feature_names_



class Num2Cyfra1(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        data = data.apply(lambda x: num2cyfra1(x))
        return data  # check what form this is in



def contains_num(x):
    if re.search('^[0-9]+.*?\s\w', str(x)):
        return 1
    else:
        return 0


# kinda awkward because you're creating it from another dataset...so you're depending on that one to be complete
def create_dataset(datafile = settings.get_dataset_path(name), docid=False): #datafile=settings.dataset_path + 'heading_id_intext_dataset.csv', docid=False):
    sourcefile = settings.get_dataset_path('marginal_lines')
    df = pd.read_csv(sourcefile, dtype={'DocID': int, 'PageNum': int, 'LineNum': int, 'Heading': int})

    if docid:
        df = df.loc[df['DocID'] == float(docid)]
    # remove ContainsTab, ContainsPage
    df = df.drop(['ContainsTab', 'ContainsPage'], axis=1)
    # remove rows with Marginal == 1 or 2. then remove marginal column
    df = df.loc[(df.Marginal == 0) | (df.Marginal != df.Marginal)]
    df = df.drop(['Marginal'], axis=1)
    # find ALL the toc pages and remove their lines from the dataset
    # find ALL the fig pages and remove their lines from the dataset
    toc_dataset = pd.read_csv(settings.get_dataset_path('toc'))
   # fig_dataset = pd.read_csv(settings.get_dataset_path('fig'))
    tocs = toc_dataset.loc[toc_dataset.TOCPage == 1]
    #figs = fig_dataset.loc[fig_dataset.FigPage == 1]
    toc_tuples = [(id, page) for id, page in zip(tocs.DocID, tocs.PageNum)]
    #fig_tuples = [(id, page) for id, page in zip(figs.DocID, figs.PageNum)]
    to_drop = []
    for i, row in df.iterrows():
        if (row.DocID, row.PageNum) in toc_tuples: #or (row.DocID, row.PageNum) in fig_tuples:
            to_drop.append(i)
    df = df.drop(index=to_drop)

    # update contains num to just re.search('[0-9]+')
    df['ContainsNum'] = df.Text.apply(lambda x: contains_num(x))
    # add column: line word count
    df.dropna(subset=['Text'], inplace=True)    # remove nans
    df['WordCount'] = df.Text.apply(lambda x: len(x.split()))

    proc_df = pd.read_csv(settings.get_dataset_path('proc_heading_id_toc'))
    proc_head_df = proc_df.loc[proc_df.Heading > 0]  # works with None type? no, but works with NaN and it should be that
    proc_head_df['Text'] = proc_head_df.apply(lambda x: str(x.SectionPrefix) + ' ' + x.SectionText, axis=1)
    series_mh = pd.Series()
    series_mt = pd.Series()
    series_mi = pd.Series()

    for id in df.DocID.unique():
        doc_toc = proc_head_df.loc[proc_head_df.DocID == float(id)]
        df_doc = df.loc[df.DocID == float(id)]
        matches_heading, matches_type, matches_i = compare_lines2headings(df_doc.Text, doc_toc)
        print(len(matches_heading) == df_doc.shape[0], id)
        series_mh = series_mh.append(pd.Series(matches_heading), ignore_index=True)
        series_mt = series_mt.append(pd.Series(matches_type), ignore_index=True)
        series_mi = series_mi.append(pd.Series(matches_i), ignore_index=True)

    df['MatchesHeading'], df['MatchesType'], df['MatchesI'] = series_mh, series_mt, series_mi
    df['TagMethod'] = None
    df[y_column] = None
    prev_dataset = settings.dataset_path + 'heading_id_intext_dataset.csv'
    df = mlh.add_legacy_y(prev_dataset, df, y_column, line=True)
    # if os.path.exists(prev_dataset):
    #     prev = pd.read_csv(prev_dataset, dtype={'DocID': int, 'PageNum': int, 'LineNum': int, 'Heading': int})
    #
    #     #df['Heading'].loc[(prev['DocID'] == df['DocID']) & (prev['PageNum'] == df['PageNum']) & (prev['LineNum'] == df['LineNum'])] = prev['Heading']
    #     df[y_column] = df.apply(lambda x: mlh.assign_y(x, prev, y_column, line=True), axis=1)
    #     df['TagMethod'].loc[df[y_column] == df[y_column]] = "legacy"

    if not docid:
        df.to_csv(datafile, index=False)
    #df['Heading'] = 0
    return df


# def assign_y(x, prev):
#     d, p, l = int(x['DocID']), int(x['PageNum']), int(x['LineNum']) - 1  # prev dataset has linenum starting at 0 >:[
#     y = prev['Heading'].loc[(prev['DocID'] == d) & (prev['PageNum'] == p) & (prev['LineNum'] == l)]
#     if len(y) == 0:
#         return None
#     elif len(y) == 1:
#         return y.values[0]
#     else:
#         print("more rows than 1")  # very possible now that multiple toc pages are possible and legacy doesn't have pagenum to compare against
#         print(y.values)

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


# def edit_dataset(dataset=settings.dataset_path + 'heading_id_intext_dataset.csv'):
#     df = pd.read_csv(dataset)
#     #df['WordCount'] = df.Text.apply(lambda x: len(x.split()))
#     # need to reference heading_id_dataset.csv: DocId, LineText, Heading[0, 1, 2]
#     toc_df = pd.read_csv(settings.dataset_path + 'processed_heading_id_dataset.csv')#, columns=['DocID', 'LineText', 'Heading'])
#     toc_head_df = toc_df.loc[toc_df.Heading > 0]
#     #df['MatchesHeading'], df['MatchesType'], df['MatchesI'] = pd.Series([]), pd.Series([]), pd.Series([])
#     toc_head_df['Text'] = toc_head_df.apply(lambda x: str(x.SectionPrefix) + ' ' + x.SectionText, axis=1)
#     series_mh = pd.Series()
#     series_mt = pd.Series()
#     series_mi = pd.Series()
#
#     for docid in df.DocID.unique():
#         doc_toc = toc_head_df.loc[toc_head_df.DocID == float(docid)]
#         df_doc = df.loc[df.DocID == float(docid)]
#         matches_heading, matches_type, matches_i = compare_lines2headings(df_doc.Text, doc_toc)
#         print(len(matches_heading) == df_doc.shape[0], docid)
#         series_mh = series_mh.append(pd.Series(matches_heading), ignore_index=True)
#         series_mt = series_mt.append(pd.Series(matches_type), ignore_index=True)
#         series_mi = series_mi.append(pd.Series(matches_i), ignore_index=True)
#
#     df['MatchesHeading'], df['MatchesType'], df['MatchesI'] = series_mh, series_mt, series_mi
#     df.to_csv(dataset, index=False)


#def rm_empty(string):
#    return re.sub('^(|\s+)$', np.nan, str(string))

#
# def data_prep(df, y=False):  # not currently called, need to include it somehow in al_data_prep
# #    df = df.apply(lambda x: rm_empty(x))
# #    df = df.dropna()
#     original_cols = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum', 'Text', 'Words2Width', 'WordsWidth', 'Width',
#                      'Height', 'Left','Top', 'ContainsNum', 'Centrality', 'Heading', 'WordCount', 'MatchesHeading','MatchesType', 'MatchesI']
#
#     df = pd.DataFrame(df, columns=original_cols)  # ordering as the fit, to not cause error in ColumnTranformer
#     if y:
#         y = y_column
#     return mlh.data_prep(df, y, limit_cols)
#     # X = df.drop(columns=) #'MatchesType',
#     # if limit_cols:
#     #     X = X.drop(columns=limit_cols)
#     # if y:
#     #     Y = df.Heading
#     #     return X, Y
#     # else:
#     #     return X


def train(n_queries=10, mode=settings.dataset_version):  #datafile=settings.get_dataset_path('heading_id_intext'), model_file=settings.get_model_path('heading_id_intext'),
    datafile = settings.get_dataset_path(name, mode)
    model_file = settings.get_model_path(name, mode)
    data = pd.read_csv(datafile)
    if 'no_toc' in model_file:
        limit_cols.extend(['MatchesHeading', 'MatchesType'])

    estimator = Pipeline([
        ('text', ColumnTransformer([
            ('cnb', Text2CNBPrediction(), 1)  # 1 HAS TO BE 'TEXT'. changing it to int bc AL uses np arrays
        ], remainder="passthrough")),
        ('forest', RandomForestClassifier())
    ], verbose=True)

    accuracy, learner = active_learning.train(data, y_column, n_queries, estimator, datafile, limit_cols)

    print(accuracy)
    with open(model_file, "wb") as file:
        pickle.dump(learner, file)
    print("End of training stage. Re-run to train again")


# def classify_line(data, mode): #model_file=settings.heading_id_intext_model_file):
#     if not os.path.exists(settings.get_model_path(name, mode)):
#         train(n_queries=0, datafile=settings.get_dataset_path(name, mode), model_file=settings.get_model_path(name, mode))
#     # with open(model_file, "rb") as file:
#     #     model = pickle.load(file)
#     # data = data_prep(data)
#     # pred = model.predict(data)
#     return mlh.classify(data, name, mode=mode, limit_cols=limit_cols)


def get_headings_intext(data, toc_page=True, mode=settings.dataset_version):
    if not toc_page:
        #pred = mlh.classify(data, name, limit_cols, mode)
        return mlh.get_classified(data, name + '_no_toc', y_column, limit_cols, mode=mode)
    else:
        headings =  mlh.get_classified(data, name, y_column, limit_cols, mode=mode)
        return headings.loc[headings.MatchesHeading > 0]
        #pred = classify_line(data, model_file=settings.get_model_path(name, mode))
    # data[y_column] = pred
    # headings = data.loc[pred > 0]
    #
    # if toc_page:
    #     return headings.loc[headings.MatchesHeading > 0]
    # else:
    #     return headings


if __name__ == '__main__':
    data_path = settings.dataset_path + 'heading_id_intext_dataset.csv'
    #data = pd.read_csv(data_path)
    #create_dataset()
    #train()
    #edit_dataset(data_path)
    data = pd.read_csv(data_path)
    get_headings_intext(data, True, mode=settings.production)
    # model_files = [settings.get_model_path('heading_id_intext'), settings.get_model_path('heading_id_intext_no_toc')]
    # for file in model_files:
    #     train(model_file=file)
        # preds = classify(data, model_file=file)
        # x = 0
        # for i, row in data.iterrows():
        #     if preds[i] != row.Heading:
        #         print(row.DocID, '\t', row.PageNum, ',', row.LineNum, '\t', row.Text, ' | ', row.Heading, ' | ', preds[i])
        #         x += 1
        # print('Wrong classifications: ', x)