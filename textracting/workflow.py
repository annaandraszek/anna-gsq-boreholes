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
from heading_id_intext import Text2CNBPrediction, Num2Cyfra1, num2cyfra1  # have to load these to load the model

if __name__ == '__main__':
    #docids = textloading.get_reportid_sample(5)
    docids = ['81317']
    print(docids)
    for docid in docids:
        #textmain.textract(docid, features=['TABLES', 'FORMS'])
        #texttransforming.clean_and_restruct(docid)
        report = search_report.Report(docid)  # need every ml method here to be able to create a dataset with an unseen report
        #search_report.draw_report(report)
        search_report.bookmark_report(report)
        search_report.save_report_sections(report)
