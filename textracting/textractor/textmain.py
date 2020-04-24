## @package textracting
#@file
#Main file for managing textract functions

from textracting import texttransforming, textracting, textsettings
from textracting.textracting import TextBasedFileException


def textract(docid: str, features: list, training=True, report_num=1):
    """
    Wrapper function for running Textract on a file in S3 and saving the response
    Features can equal to any subset of ['TABLES', 'FORMS']
    """
    textracting.report2textract(docid, features=features, write_bucket=textsettings.read_bucket, training=training, report_num=report_num)


if __name__ == "__main__":
    # Specify whatever DocIDs in docs and will get Textract response and clean it up for you
    pre = 'cr_' # 'smaller_'
    docs = ['100000'] #['32730', '44448', '37802', '2646', '44603']  #['30281']
    for doc_path in docs:
        docid = doc_path # pre +
        #documentName = pre + doc_path + '_1.pdf' #'.pdf'
        features=['TABLES', 'FORMS']
        try:
            textract(docid, features)
        except TextBasedFileException as e:
            print(e)
            continue
        res = texttransforming.clean_and_restruct(docid, save=False) # pagelineinfo -> cleanpage -> restructpageinfo
        print(res)


