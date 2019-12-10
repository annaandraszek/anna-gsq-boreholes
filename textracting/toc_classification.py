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


if __name__ == "__main__":
    #reports = get_reportid_sample()
    rfs = glob.glob('training/QDEX/*')
    report_ids = set([r.rsplit('\\')[-1] for r in rfs])
    #reports = []
    #download_reports(reports, 'training/')

    for report in report_ids:
        upload_to_my_bucket(report)
        doc2data(report)

