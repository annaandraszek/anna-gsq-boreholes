from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import settings
import pickle
import eli5
# dataset is edited down version of heading_id_intext.csv, and annotated


def data_prep(df, y=False):
    X = df.Text
    if y:
        y = df.HeadingClass
        return X, y
    return X


def train(data, model_file=settings.heading_classification_model_file):
    X, Y = data_prep(data, y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
    clf = Pipeline([('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1,2))),#(token_pattern=r'([a-zA-Z]|[0-9])+')),
                    ('clf', ComplementNB(norm=True))])

    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #weights = eli5.formatters.as_dataframe.explain_weights_df(clf, feature_names=clf['tfidf'].get_feature_names(), top=10, target_names=y_test)
    #print(weights)
    #prediction = eli5.formatters.as_dataframe.explain_prediction_df(clf, X_test[0], feature_names=clf['tfidf'].get_feature_names(), target_names=y_test)
    #print(prediction)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    report = classification_report(Y, clf.predict(X))
    print(report)
    with open(settings.result_path + 'heading_classification_CNB_report.txt', "w") as r:
        r.write(report)
    with open(model_file, "wb") as file:
        pickle.dump(clf, file)


def predict(inputs):
    if isinstance(inputs, str):
        inputs = [inputs]
    with open(settings.heading_classification_model_file, "rb") as file:
        model = pickle.load(file)
    pred = model.predict(inputs)
    proba = model.predict_proba(inputs)
    return pred, proba#, model


def classify(data):
    X = data_prep(data)
    pred, proba = predict(X)
    data['HeadingType'] = pred
    return data


if __name__ == '__main__':
    dataset = settings.dataset_path + 'heading_classification_dataset.csv'
    df = pd.read_csv(dataset)
    train(df)
    preds = predict(df.Text)
    #print(preds)

