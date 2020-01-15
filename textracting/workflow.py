# download reports from s3 gsq-staging ?

# run textracting on reports, saving various json, csv, txt files

# with small number of noisy reports, remove noise with heuristics, (and potentially use dataset and model in the future)
    # cleanpage s are created from pagelineinfo. from these, restructpagelines are be created

# create toc dataset from files and train
# todo create figure dataset and train figure page classifier

# todo create heading recognition dataset and train
# todo create heading identification dataset and train


import textmain
import textloading
import search_report
import texttransforming

if __name__ == '__main__':
    #docids = textloading.get_reportid_sample(2)
    docids = ['24352', '28184', '30281']
    print(docids)
    #textmain.textract_many(docids, features=['TABLES', 'FORMS'])
    for docid in docids:
        texttransforming.clean_and_restruct(docid)
        report = search_report.Report(docid)  # need every ml method here to be able to create a dataset with an unseen report
        search_report.draw_report(report)
        search_report.bookmark_report(report)
