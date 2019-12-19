# finding page(s) in a document that is a table of contents

# process
# a) classifier training
    # 1. create a training set
        # - download mass of documents from [BHP] [1990s+] [open] from s3
        # - upload to my s3 bucket, gsq-ml
        # - for each document,
            # put it through textract, getting back the json
            # from the json, extract the page blocks into a new list
            # from the json, also extract the page children (lines) text and position, into a new structure
            # for each page, find if "table of contents" is in the page text, and if so, give its position to the pages structure
            # create a datframe or np array which will be exported to csv which contains:
                # [page number] [number of children] ["table of contents" in children, position?] [table of contents?]
            # manually tag if the page is a ToC or not
        # (training set should not need pruning to balance ToC/non-ToC pages, as the imbalance will be true for real documents as well
    # 2. get a model and train it
        # from scikit-learn, choose SVM or decision tree
            # SVM: will need data normalisation
            # decision tree: will need all values full
    # 3. test model
        # with f1 score
    # 4. report results

# b) classifier use
    # 1. use with real documents
        # pipeline:
            # input document name on s3 to download, or input local document name, or input json name to not run textract

import pandas as pd
import numpy as np
import random
from s3_downloading import download_reports
from textmain import file2doc
import boto3
import glob
import img2pdf
import settings
import json
from sklearn import tree
import sklearn
import pickle
import os


def get_reportid_sample():
    random.seed(19)
    reps = pd.read_excel('../investigations/QDEX_reports_BHP.xlsx')
    rs = random.sample(list(reps.REPNO), 50)
    return rs


def doc2data(file_id):
    file = settings.get_report_name(file_id, file_extension=True)
    pageinfo, pagelines = file2doc(file, bucket='gsq-ml', features=['TABLES', 'FORMS'], pageinfo=True, ret=True)
    #print("pageinfo: \n" + pageinfo)
    #print("pagelines: \n" + pagelines)


def upload_to_my_bucket(file_id):
    s3 = boto3.client('s3')
    bucketname = 'gsq-ml'
    if check_if_obj_exists(s3, settings.get_report_name(file_id, file_extension=True), bucketname):
        return -1
    fpath = settings.get_report_name(file_id, local_path=True)
    report = glob.glob(fpath + ".*")[0]
    if ".tif" in report:
        pdf_report = settings.get_report_name(file_id, local_path=True, file_extension=True)
        with open(pdf_report, "wb") as f:
            f.write(img2pdf.convert(report))
    else:
        pdf_report = report

    with open(pdf_report, "rb") as f:
        objname = settings.get_report_name(file_id, file_extension=True)
        s3.upload_fileobj(f, bucketname, objname)
    return 0


def check_if_obj_exists(client, key, bucket):
    response = client.list_objects_v2(
        Bucket=bucket,
        Prefix=key,
    )
    for obj in response.get('Contents', []):
        if obj['Key'] == key:
            return True
        else:
            return False


def create_dataset():
    df = pd.DataFrame(columns=['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'TOCPage'])
    pageinfos = sorted(glob.glob('training/pageinfo/*'))
    pagelines = sorted(glob.glob('training/pagelines/*'))

    for pagesinfo, pageslines, i in zip(pageinfos, pagelines, range(len(pageinfos))):
        pi = json.load(open(pagesinfo))
        pl = json.load(open(pageslines))
        docset = np.zeros((len(pi.items()), 6))
        docid = pagesinfo.split('\\')[-1].replace('_1_pageinfo.json', '')

        for info, lines, j, in zip(pi.items(), pl.items(), range(len(pi.items()))):
            toc = 0
            c = 0
            for line in lines[1]:
                if 'contents' in line.lower():
                    c = 1
                    if 'table of contents' in line.lower():
                        toc = 1

            docset[j] = np.array([docid.strip('cr_'), info[1]['Page'], len(lines[1]), toc, c, 0])
        pgdf = pd.DataFrame(data=docset, columns=['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord', 'TOCPage'])
        df = df.append(pgdf, ignore_index=True)
    return df


def data_prep(data, y=False):
    #data = data.drop(['Comments'], axis=1)
    data = data.dropna()
    X = data[['PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord']]
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
    with open(settings.toc_tree_model_file, "wb") as file:
        pickle.dump(clf, file)


def classify_page(data):
    if not os.path.exists(settings.toc_tree_model_file):
        train(data)
    with open(settings.toc_tree_model_file, "rb") as file:
        model = pickle.load(file)
    data = data_prep(data)
    pred = model.predict(data)
    return pred


def get_toc_pages(docid):
    data_file = settings.production_path + docid + '_toc_dataset.csv'
    df = pd.read_csv(data_file)
    classes = classify_page(df)
    mask = np.array([True if i==1 else False for i in classes])
    toc_pages = df[mask]
    return toc_pages


if __name__ == "__main__":
    #reports = get_reportid_sample()
    # rfs = glob.glob('training/QDEX/*')
    # report_ids = set([r.rsplit('\\')[-1] for r in rfs])
    # #reports = []
    # #download_reports(reports, 'training/')
    #
    # for report in report_ids:
    #     res = upload_to_my_bucket(report)
    #     if res != 0:
    #         continue
    #     doc2data(report)

    # dataset = create_dataset()
    # dataset.to_csv('toc_dataset.csv', index=False)
    # print(dataset)

    toc_pages = get_toc_pages()
    print(toc_pages)
