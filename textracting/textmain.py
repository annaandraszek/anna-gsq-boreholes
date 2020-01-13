
# High level management of textract functions

import textracting
import texttransforming


if __name__ == "__main__":
    s3BucketName = 'gsq-staging'
    pre = 'cr_' # 'smaller_'
    docs = ['30281']
    for doc_path in docs:
        docid = doc_path # pre +
        documentName = pre + doc_path + '_1.pdf' #'.pdf'
        features=['TABLES', 'FORMS']
        #textracting.report2textract(documentName, s3BucketName, features)
        res = texttransforming.clean_and_restruct(docid, save=False) # pagelineinfo -> cleanpage -> restructpageinfo
        print(res)