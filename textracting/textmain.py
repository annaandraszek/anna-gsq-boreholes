## @file
#Main file for managing textract functions

import textracting
import texttransforming


# textract a report with docid in s3
# features can = ['TABLES', 'FORMS']
def textract(docid, features):
    #documentName = 'cr_' + docid + '_1.pdf'  # '.pdf'
    textracting.report2textract(docid, features=features, bucket='gsq-horizon')
    #texttransforming.clean_and_restruct(docid, save=True)  # pagelineinfo -> cleanpage -> restructpageinfo


def textract_many(docids, features):
    for docid in docids:
        textract(docid, features)


if __name__ == "__main__":
    s3BucketName = 'gsq-horizon'
    pre = 'cr_' # 'smaller_'
    docs = ['30281']
    for doc_path in docs:
        docid = doc_path # pre +
        #documentName = pre + doc_path + '_1.pdf' #'.pdf'
        features=['TABLES', 'FORMS']
        textracting.report2textract(docid, s3BucketName, features)
        res = texttransforming.clean_and_restruct(docid, save=False) # pagelineinfo -> cleanpage -> restructpageinfo
        print(res)