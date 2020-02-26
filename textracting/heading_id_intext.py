## @file
# Module functions for finding headings in the report text

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import textdistance
import numpy as np
import active_learning
import machine_learning_helper as mlh
import heading_id_toc

name = 'heading_id_intext'
y_column = 'Heading'
limit_cols = ['DocID', 'LineNum', 'WordsWidth', 'NormedLineNum', 'Top', 'Heading', 'Centrality', 'MatchesI']
include_cols = ['PageNum', 'Text', 'Words2Width', 'Width', 'Height', 'Left', 'ContainsNum', 'WordCount', 'MatchesHeading', 'MatchesType']


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
        self.y_ = y
        return self

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
        data = data.apply(lambda x: heading_id_toc.num2cyfra1(x, remove_words=False))
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
    if not docid:
        df.to_csv(datafile, index=False)
    #df['Heading'] = 0
    return df


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
            hd = heading.Text.lower()
            hd, _ = heading_id_toc.split_pagenum(hd)
            hd_words = hd.split()
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


def get_headings_intext(data, toc_page=True, mode=settings.dataset_version):
    if not toc_page:
        #pred = mlh.classify(data, name, limit_cols, mode)
        return mlh.get_classified(data, name + '_no_toc', y_column, limit_cols, mode=mode)
    else:
        headings =  mlh.get_classified(data, name, y_column, limit_cols, mode=mode)

        return headings.loc[headings.MatchesHeading > 0]


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



#
# def num2cyfra1(string):
#     s = ''
#     prev_c = ''
#     i = 1
#     for c in string:
#         if re.match(r'[0-9]', c):
#             if prev_c != 'num':
#                 if c != '0':  # to stop eg. 1.0 being tagged as cyfra1 punkt cyfra2 like a subheading
#                     s += ' cyfra' + str(i) + ' '
#                     i += 1
#                     prev_c = 'num'
#         elif c == '.':
#             s += ' punkt '
#             prev_c = '.'
#         else:
#             s+= c
#             #prev_c = 'char'
#     return s
