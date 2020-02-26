## @file
#Main file for managing textract functions

from textracting import textracting
from textracting import texttransforming
from textracting import textsettings


def textract(docid: str, features: list):
    """
    Wrapper function for running Textract on a file in S3 and saving the response
    Features can equal to any subset of ['TABLES', 'FORMS']
    """
    textracting.report2textract(docid, features=features, write_bucket=textsettings.read_bucket)


def textract_many(docids: list, features: list):
    """ Run Textract on many DocIDs in one go"""
    for docid in docids:
        textract(docid, features)


if __name__ == "__main__":
    # Specify whatever DocIDs in docs and will get Textract response and clean it up for you
    pre = 'cr_' # 'smaller_'
    docs = ['30281']
    for doc_path in docs:
        docid = doc_path # pre +
        #documentName = pre + doc_path + '_1.pdf' #'.pdf'
        features=['TABLES', 'FORMS']
        textract(docid, features)
        res = texttransforming.clean_and_restruct(docid, save=False) # pagelineinfo -> cleanpage -> restructpageinfo
        print(res)