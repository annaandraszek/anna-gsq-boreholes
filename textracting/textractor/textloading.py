## @package textractor
#@file
# Pre-textract functions

import random
import pandas as pd
import boto3
import os
#from datetime import timedelta
import settings
from textractor import textsettings


def download_report(fname, dest):
    s3 = boto3.resource('s3')
    s3.Bucket(textsettings.read_bucket).download_file(fname, dest)


def get_reportid_sample(num=50, submitter=None, rtype_exclude=None, cutoffdate=pd.Timestamp(1990, 1, 1), rtitle_exclude=None, rtype_include=None, all=False):
    #random.seed(19)
    #reps = edit_reports_xlsx()
    reps = pd.read_excel('../../investigations/QDEX_export_v2.xlsx')#_reports_BHP.xlsx')
    reps = reps.loc[reps.RSTATUS.str.contains('C') == False]  # exclude confidential
    reps.REPNO = reps.REPNO.astype(int)

    if all:
        return [str(r) for r in reps.REPNO.values]

    if cutoffdate:
        reps = reps.loc[reps.REPDATE > cutoffdate]
    if submitter:
        reps = reps.loc[reps.SUBMITBY.str.contains(submitter)]

    if isinstance(rtype_include, list):
        for type in rtype_include:
            reps = reps.loc[reps.RTYPE.str.contains(type)]
    elif isinstance(rtype_exclude, list):
        for type in rtype_exclude:
            reps = reps.loc[not reps.RTYPE.str.contains(type)]
    if isinstance(rtitle_exclude, list):
        for title in rtitle_exclude:
            reps = reps.loc[not reps.RTITLE.str.contains(title)]
    rs = random.sample(list(reps.REPNO), num)
    return [str(r) for r in rs]


def find_file(docid, report_num=1):  # for finding the file type of a report
    file_pre = settings.get_s3_location(docid, format=None, report_num=report_num)
    client = boto3.client('s3')
    #my_bucket = s3.Bucket('gsq-staging/QDEX')
    files = client.list_objects_v2(Bucket=textsettings.read_bucket, Prefix=file_pre)
    if not 'Contents' in files.keys():
        raise FileNotFoundError
    for file in files['Contents']:
        if file['Key'].startswith(file_pre + '.'):
            return file['Key']


def get_subdir_contents(docid, textractable=True):
    file_pre = settings.get_s3_subdir(docid)
    client = boto3.client('s3')
    #my_bucket = s3.Bucket('gsq-staging/QDEX')
    files = client.list_objects_v2(Bucket=textsettings.read_bucket, Prefix=file_pre)
    if textractable:
        textractablefiles = []
        for file in files['Contents']:
            if file['Key'].endswith('.pdf') or file['Key'].endswith('.tif'):
                textractablefiles.append(file['Key'])
        return textractablefiles
    else:
        return [f['Key'] for f in files['Contents']]


def get_report_nums_from_subdir(docid, textractable=True):
    files = get_subdir_contents(docid, textractable=textractable)
    nums = []
    for file in files:
        end = file.rsplit('_', 1)[1]
        num = end.split('.')[0]
        nums.append(num)
    return nums