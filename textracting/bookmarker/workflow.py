## @file
# Main file for bookmarking report from the command line

import sys
sys.path.append('../')
from textracting import textmain, textloading, texttransforming, textracting
from bookmarker import search_report
#from heading_id_intext import Text2CNBPrediction, Num2Cyfra1, num2cyfra1  # have to load these to load the model
import os
import settings
import pandas as pd
import time
import csv
import datetime
import argparse
#import numpy as np
import warnings
import pickle as pkl

num_sample = 20
cutoffdate = None
rtype_exclude = None #'WELCOM'
bookmark = False
training = False
all = True
special_mode = "no" #"testing"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--id", help="report IDs to bookmark", nargs='+')  # list type
    parser.add_argument("-s", "--sample", help='number of reports to sample', nargs='?', default=num_sample, type=int) # can store just sample mode, or sample num
    parser.add_argument("--save", help="use to save sample arguments as default", action='store_true')
    parser.add_argument("-d", "--cutoffdate", help="cutoff year for reports to be no older than", type=int)
    parser.add_argument("--extype", help="report types to exclude. must match report type code eg. WELCOM for Well Completion Report", nargs='+') # have a more verbore help that gives all the codes?
    parser.add_argument("--extitle", help="report titles containing these phrases to exclude", nargs='+') # have a more verbore help that gives all the codes?
    parser.add_argument("--intype", help="report types to include", nargs="+")
    parser.add_argument("-f", "--force", help="force report to be processed even if already has been", action='store_true')
    args = parser.parse_args()

    var_file = 'bookmarker_vars.pkl'
    if os.path.exists(var_file):
        with open(var_file, "rb") as f:
            num_sample, cutoffdate, rtype_exclude, rtitle_exclude = pkl.load(f)
        print("Loaded vars; num_sample: ", num_sample, ' cutoffdate: ', cutoffdate, ' rtype_exclude: ', rtype_exclude, ' rtitle_exclude: ', rtitle_exclude)
    else:
        num_sample = 20
        cutoffdate = None
        rtype_exclude = None
        rtitle_exclude = None

    not_exit = True
    while not_exit:
        mode = 'sample'  # default behaviour is random sampling
        if args:  # always going to be true
            warnings.filterwarnings("ignore")
        if args.save:
            mode = 'save'
            with open(var_file, "wb") as f:
                if args.sample:
                    num_sample = args.sample
                if args.cutoffdate:
                    cutoffdate = pd.Timestamp(args.cutoffdate, 1, 1)
                if args.extype:
                    rtype_exclude = args.extype
                if args.extitle:
                    rtitle_exclude = args.extitle
                pkl.dump([num_sample, cutoffdate, rtype_exclude, rtitle_exclude], f)
            print("Saved vars; num_sample: ", num_sample, ' cutoffdate: ', cutoffdate, ' rtype_exclude: ',
                  rtype_exclude, ' rtitle_exclude: ', rtitle_exclude)

        if args.sample:
            if args.intype:
                rtype_include = args.intype
            else:
                rtype_include = None
            if mode != "save":  # case: saving num_sample but not in sample mode
                mode = 'sample'
                num_sample = args.sample
                if args.cutoffdate:
                    cutoffdate = pd.Timestamp(args.cutoffdate, 1, 1)
                if args.extype:
                    rtype_exclude = args.extype
                if args.extitle:
                    rtitle_exclude = args.extitle

        if args.id:
            docids = args.id
            mode = 'given'

        if mode == "sample" or mode == "given" or special_mode == "testing":
            if special_mode == "testing":
                print("Running in testing mode")
                docids = ['6']

            elif mode == 'sample':
                if rtype_include:
                    print("Running in sample mode. Num samples: " + str(num_sample) + "Including type: " + str(rtype_include))
                    docids = textloading.get_reportid_sample(num=num_sample, rtype_include=rtype_include)
                else:
                    print("Running in sample mode. Num samples: " + str(num_sample) + " Cutoff date: " + str(cutoffdate) +
                      " Excluding: " + str(rtype_exclude) + " " + str(rtitle_exclude))
                    docids = textloading.get_reportid_sample(num=num_sample, cutoffdate=cutoffdate, rtype_exclude=rtype_exclude, rtitle_exclude=rtitle_exclude)

            elif mode == 'given':
                print("Running in 'given' mode")

            #training_folders = os.walk('training/QDEX/')
            #training_docids = [x[0].split('\\')[-1] for x in training_folders]

            #docids = ['15042', '41275', '4639', '48670', '593', '3051', '24357', '15568', '68677', '48897', '36490', '5261', '44433'] #'41568', '41982', '10189', '102109', '43758', '105472', '48907'
            print("Report IDs to bookmark: ",  docids)
            log_file = 'bookmarker_log.csv'
            # log file cols = report_id, time2textract, time2ml, toc, time_run
            if not os.path.exists(log_file):
                with open(log_file, "w", newline='') as log:
                    writer = csv.writer(log)
                    writer.writerow(['report_id', 'time2textract', 'time2ml', 'toc', 'time_run'])

            for docid in docids:
                # all the below checks also need to check if the --force arg is True, which would overrule their skip
                # check if textract needs to be run or if fulljson already exists
                if all:
                    nums = textracting.get_report_nums_from_subdir(docid, textractable=True)
                else:
                    nums = [1]
                print('Nums: ', nums)
                for num in nums:
                    if not (os.path.exists(settings.get_full_json_file(docid, training=training, report_num=num))) and (not args.force):
                        textract_start = time.time()
                        try:
                            textmain.textract(docid, features=['TABLES', 'FORMS'], training=training, report_num=num)
                        except FileNotFoundError:
                            print("Report file", docid, "_", str(num), "doesn't exist in S3")
                            continue
                        textract_end = time.time()
                        textract_time = textract_end - textract_start
                        print("Time to textract: " + str(docid) + "_" + str(num) + " " + "{0:.2f}".format(textract_time) + " seconds")
                    else:
                        print("Report ", docid, "_", str(num),  " already textracted")
                        textract_time = 0

                    # check if clean and restruct needs to be run or if restructpageinfo alredy exists
                    if (not os.path.exists(settings.get_restructpageinfo_file(docid, training=training, report_num=num)) and (not args.force)):
                        texttransforming.clean_and_restruct(docid, save=True, training=training, report_num=num)
                    else: print("Report ", docid, "_", str(num), " already cleaned and reconstructed")
                    # check if search report, bookmark report, needs to be run or if bookmarked pdf already exists
                if bookmark:
                    if (not os.path.exists(settings.get_bookmarked_file(docid))) and (not args.force):
                        ml_start = time.time()
                        try:
                            report = search_report.Report(docid)  # need every ml method here to be able to create a dataset with an unseen report
                        except ValueError:
                            continue
                        #search_report.draw_report(report)
                        search_report.bookmark_report(report)
                        # check if needs to be run or if sections word doc already exists
                        search_report.save_report_sections(report)
                        search_report.report2json(report)

                        ml_end = time.time()
                        ml_time = ml_end - ml_start
                        print("Time to ML, bookmark, export to text: " + "{0:.2f}".format(ml_time) + " seconds")
                        print("COMPLETED BOOKMARKING " + docid + ", total time: " + "{0:.2f}".format(
                            ml_time + textract_time) + " seconds")
                        toc_exists = True if report.toc_page else False
                        bookmark_time = datetime.datetime.now()
                with open(log_file, 'a', newline='') as log:
                    writer = csv.writer(log)
                    writer.writerow([int(docid), textract_time, None, None])#, #bookmark_time]) #ml_time, toc_exists, bookmark_time])
                #else: print("Report already bookmarked")

        cont = input("Run again?")
        if 'n' in cont:
            not_exit = False
        else:
            new_args = input("Enter new args: ")
            args = parser.parse_args(new_args.split())

