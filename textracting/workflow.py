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
import os
import settings
import pandas as pd
import time


if __name__ == '__main__':
    #docids = textloading.get_reportid_sample(num=20, cutoffdate=pd.Timestamp(2000, 1, 1), rtype_exclude='WELCOM')
    training_folders = os.walk('training/QDEX/')
    training_docids = [x[0].split('\\')[-1] for x in training_folders]
    #docids = #['37038', '34597', '52161', '33201', '70562'] # '45198', '80507', '75082', '41583', '76890', '70158', '52182', '40124', '75275', '42133', '54223', '54437', '40826', '31743', '38400',
    docids = ['31743', '37038', '42133', '52182']

    print(docids)
    for docid in docids:
        if docid not in training_docids:
            textract_start = time.time()
            # try:
            #     textmain.textract(docid, features=['TABLES', 'FORMS'])
            # except FileNotFoundError:
            #     print ("Report file doesn't exist in S3")
            #     continue
            textract_end = time.time()
            textract_time = textract_end - textract_start
            print("Time to textract: " + str(docid) + " " + str(textract_time))
            ml_start = time.time()
            texttransforming.clean_and_restruct(docid)
            report = search_report.Report(docid)  # need every ml method here to be able to create a dataset with an unseen report
            #search_report.draw_report(report)
            search_report.bookmark_report(report)
            search_report.save_report_sections(report)
            ml_end = time.time()
            ml_time = ml_end - ml_start
            print("Time to ML, bookmark, export to text: " + str(ml_time))
            print("COMPLETED BOOKMARKING " + docid + ", total time: " + str(ml_time + textract_time))

