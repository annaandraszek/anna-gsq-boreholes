## @file
# Main file for bookmarking report from the command line. This is the only needed executable.
# How to use:
#   If using AWS Textract: Modify variables in textractor/textsettings.py to match AWS config
#   Install all needed libraries with environment.yml file
#   Run from cmd with reference to the arguments at the top of main
#   To run on a random sample of reports: (with default variable values for sampling)
#       python workflow.py
#   To run on specific report ids, use --id and space separated IDs, eg:
#       python workflow.py --id 2646 78932 32424
#
# by Anna Andraszek

import sys

import textractor.textloading
sys.path.append('../')
from textractor import textmain, textloading, texttransforming, textracting
#from textractor.textracting import TextBasedFileException
from report import search_report
#from heading_id_intext import Text2CNBPrediction, Num2Cyfra1, num2cyfra1  # have to load these to load the model
import os
import paths
import pandas as pd
import time
import csv
import datetime
import argparse
#import numpy as np
import warnings
import pickle as pkl
from shutil import copyfile
from PIL.Image import DecompressionBombError


# default number of report IDs to sample
num_sample = 20
# earliest date a report can be published, for sampling
cutoffdate = None
# report types to exclude from sampling. must be given as codes, eg. WELCOM == Well Completion Report
rtype_exclude = None #'WELCOM'
# if bookmarking should be done on each report
bookmark = True # False
# if using textract
textract = False
# uses a different location for files depending on if True or False
training = True #False
# if all files in a report should be processed, or just the _1
all_files = True
# use if want to set report ids manually in this files instead of running from cmd
special_mode = '' #'testing' #"welcom"
# further adds to file location
extrafolder = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline which will do the following for files in S3 bucket: extract text and tables with Textract, segment report section, and save a bookmarked pdf and word files of text of that report.")
    parser.add_argument("-i", "--id", help="report IDs to bookmark", nargs='+')  # list type
    parser.add_argument("-s", "--sample", help='number of reports to sample', nargs='?', default=num_sample, type=int) # can store just sample mode, or sample num
    parser.add_argument("--save", help="use to save sample arguments as default", action='store_true')
    parser.add_argument("-d", "--cutoffdate", help="cutoff year for reports to be no older than", type=int)
    parser.add_argument("--extype", help="report types to exclude. must match report type code eg. WELCOM for Well Completion Report", nargs='+') # have a more verbore help that gives all the codes?
    parser.add_argument("--extitle", help="report titles containing these phrases to exclude", nargs='+') # have a more verbore help that gives all the codes?
    parser.add_argument("--intype", help="report types to include", nargs="+")
    parser.add_argument("-f", "--force", help="force report to be processed even if already has been", action='store_true')
    parser.add_argument("-a", "--all", help="run for ALL reports")
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

        if args.all:
            mode='all'

        if mode == "sample" or mode == "given" or special_mode == "testing" or special_mode == 'welcom':
            if special_mode == "testing":
                print("Running in testing mode")
                docids = ['95183']

            elif special_mode == 'welcom':
                print('running for welcom sample')
                refile = 'ReportID_for_Textract_processing.xlsx'
                ref = pd.read_excel(refile)
                docids = ref['ReportID'].values
                docids = [str(d) for d in docids]
                extrafolder = 'qutsample'

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

            elif mode == 'all':
                print("Running for all reports")
                docids = textloading.get_reportid_sample(all=True)

            #training_folders = os.walk('training/QDEX/')
            #training_docids = [x[0].split('\\')[-1] for x in training_folders]

            #docids = ['15042', '41275', '4639', '48670', '593', '3051', '24357', '15568', '68677', '48897', '36490', '5261', '44433'] #'41568', '41982', '10189', '102109', '43758', '105472', '48907'
            print("Report IDs: ",  docids)
            # log_file = 'bookmarker_log.csv'
            # # log file cols = report_id, time2textract, time2ml, toc, time_run
            # if not os.path.exists(log_file):
            #     with open(log_file, "w", newline='') as log:
            #         writer = csv.writer(log)
            #         writer.writerow(['report_id', 'time2textract', 'time2ml', 'toc', 'time_run'])

            for docid in docids:
                # all the below checks also need to check if the --force arg is True, which would overrule their skip
                # check if textract needs to be run or if fulljson already exists
                if all_files:
                    #if textract:
                    nums = textracting.textloading.get_report_nums_from_subdir(docid, textractable=True)  # lost permissions to this?
                    #else:
                    #    print("Don't know all file numbers, using just _1. Set textract=True for intended behaviour")
                    #    nums = ['1']#,'2']  # specific to 51800
                else:
                    nums = ['1']
                print('Nums: ', nums)
                for num in nums:
                    if not (os.path.exists(paths.get_full_json_file(docid, training=training, file_num=num))) and (not args.force):
                        textract_start = time.time()
                        try:
                            textmain.textract(docid, features=['TABLES'], training=training, report_num=num)
                        except FileNotFoundError:
                            #print("Report file", docid, "_", str(num), "doesn't exist in S3")
                            continue
                        except DecompressionBombError as e:
                            print(e)
                            continue
                        textract_end = time.time()
                        textract_time = textract_end - textract_start
                        print("Time to textract: " + str(docid) + "_" + str(num) + " " + "{0:.2f}".format(textract_time) + " seconds")
                    else:
                        print("Report ", docid, "_", str(num),  " already textracted")
                        textract_time = 0


                    # check if clean and restruct needs to be run or if restructpageinfo alredy exists
                    if (not os.path.exists(paths.get_restructpageinfo_file(docid, training=training, file_num=num)) and (not args.force)):
                        texttransforming.clean_and_restruct(docid, save=True, training=training, report_num=num)
                    else: print("Report ", docid, "_", str(num), " already cleaned and reconstructed")

                    if special_mode == 'welcom':
                        # copy json, tables, kvpairs, to extrafolder
                        jsonsrc = paths.get_full_json_file(docid, training=training, file_num=num)
                        jsondest = paths.get_full_json_file(docid, training=training, file_num=num, extrafolder=extrafolder)
                        try:
                            copyfile(jsonsrc, jsondest)
                        except FileNotFoundError as e:
                            print(e)

                        tablessrc = paths.get_tables_file(docid, training=training, file_num=num)
                        tablesdest = paths.get_tables_file(docid, training=training, file_num=num, extrafolder=extrafolder)
                        try:
                            copyfile(tablessrc, tablesdest)
                        except FileNotFoundError as e:
                            print(e)

                        kvssrc = paths.get_kvs_file(docid, training=training, file_num=num)
                        kvsdest = paths.get_kvs_file(docid, training=training, file_num=num, extrafolder=extrafolder)
                        try:
                            copyfile(kvssrc, kvsdest)
                        except FileNotFoundError as e:
                            print(e)

                        ressrc = paths.get_restructpageinfo_file(docid, training=training, file_num=num)
                        resdest = paths.get_restructpageinfo_file(docid, training=training, file_num=num, extrafolder=extrafolder)
                        try:
                            copyfile(ressrc, resdest)
                        except FileNotFoundError as e:
                            print(e)

                    # check if search report, bookmark report, needs to be run or if bookmarked pdf already exists
                    if bookmark:
                        if (not os.path.exists(paths.get_bookmarked_file(docid, filenum=num))) and (not args.force):
                            ml_start = time.time()
                            #try:
                            report = search_report.Report(docid, num)  # need every ml method here to be able to create a dataset with an unseen report
                            #except ValueError:
                            #    continue
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
                # with open(log_file, 'a', newline='') as log:
                #     writer = csv.writer(log)
                #     writer.writerow([int(docid), textract_time, None, None])#, #bookmark_time]) #ml_time, toc_exists, bookmark_time])
                #else: print("Report already bookmarked")

        cont = input("Run again?")
        if 'n' in cont:
            not_exit = False
        else:
            new_args = input("Enter new args: ")
            args = parser.parse_args(new_args.split())

