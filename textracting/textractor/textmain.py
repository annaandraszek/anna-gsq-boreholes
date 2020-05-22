## @package textractor
#@file
#Main file for managing textract functions

from textractor import texttransforming, textracting, textsettings, textloading
from textractor.textracting import TextBasedFileException
import time
import argparse
import warnings
import os
import csv
import paths

def textract(docid: str, features: list, training=True, report_num=1):
    """
    Wrapper function for running Textract on a file in S3 and saving the response
    Features can equal to any subset of ['TABLES', 'FORMS']
    """
    textracting.report2textract(docid, features=features, write_bucket=textsettings.read_bucket, training=training, report_num=report_num)


# if __name__ == "__main__":
#     # Specify whatever DocIDs in docs and will get Textract response and clean it up for you
#     pre = 'cr_' # 'smaller_'
#     docs = ['100000'] #['32730', '44448', '37802', '2646', '44603']  #['30281']
#     for doc_path in docs:
#         docid = doc_path # pre +
#         #documentName = pre + doc_path + '_1.pdf' #'.pdf'
#         features=['TABLES', 'FORMS']
#         try:
#             textract(docid, features)
#         except TextBasedFileException as e:
#             print(e)
#             continue
#         res = texttransforming.clean_and_restruct(docid, save=False) # pagelineinfo -> cleanpage -> restructpageinfo
#         print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--id", help="report IDs to bookmark", nargs='+')  # list type
    parser.add_argument("-s", "--sample", help='number of reports to sample', nargs='?', default=textsettings.num_sample, type=int) # can store just sample mode, or sample num
    parser.add_argument("-f", "--force", help="force report to be processed even if already has been", action='store_true')
    parser.add_argument("-a", "--all", help="run for ALL reports")
    args = parser.parse_args()

    not_exit = True
    while not_exit:
        mode = 'sample'  # default behaviour is random sampling
        if args:  # always going to be true
            warnings.filterwarnings("ignore")

        if not args.id:
            mode = 'sample'
            num_sample = args.sample
            print("Running in sample mode. Num samples: " + str(num_sample))
            docids = textloading.get_reportid_sample(num=num_sample)

        elif args.id:
            docids = args.id
            mode = 'given'
            print("Running for given reports IDs")

        if args.all:
            mode = "all"
            print("Running for all reports")
            docids = textloading.get_reportid_sample(all=True)

        print("Report IDs to bookmark: ",  docids)
        for docid in docids:
            # all the below checks also need to check if the --force arg is True, which would overrule their skip
            # check if textract needs to be run or if fulljson already exists
            if textsettings.all_files:
                nums = textracting.textloading.get_report_nums_from_subdir(docid, textractable=True)
            else:
                nums = [1]
            print('Nums: ', nums)
            for num in nums:
                if not (os.path.exists(paths.get_full_json_file(docid, file_num=num))) and (not args.force):
                    textract_start = time.time()
                    try:
                        textract(docid, features=['TABLES'], report_num=num)
                    except FileNotFoundError as e:
                        #print("Report file", docid, "_", str(num), "doesn't exist in S3")
                        print(e)
                        continue
                    except TextBasedFileException as e:
                         print(e)
                         continue
                    textract_end = time.time()
                    textract_time = textract_end - textract_start
                    print("Time to textract: " + str(docid) + "_" + str(num) + " " + "{0:.2f}".format(textract_time) + " seconds")
                else:
                    print("Report ", docid, "_", str(num),  " already textracted")

                # check if clean and restruct needs to be run or if restructpageinfo alredy exists
                if (not os.path.exists(paths.get_restructpageinfo_file(docid, file_num=num)) and (not args.force)):
                    texttransforming.clean_and_restruct(docid, save=True, report_num=num)
                else: print("Report ", docid, "_", str(num), " already cleaned and reconstructed")

        cont = input("Run again?")
        if 'n' in cont:
            not_exit = False
        else:
            new_args = input("Enter new args: ")
            args = parser.parse_args(new_args.split())
