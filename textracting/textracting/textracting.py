## @package textracting
#@file
# Running textract on documents in S3

import boto3
#import time
import json
#import textmain
from textracting import texttransforming, textloading, textsettings
import settings
import re
import img2pdf
import os
import time


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


def startJob(s3BucketName, objectName, features=None):
    response = None
    client = boto3.client('textract', region_name=textsettings.region)
    response = client.start_document_analysis(
        DocumentLocation={
            'S3Object': {
                'Bucket': s3BucketName,
                'Name': objectName
            }
        },
        FeatureTypes=features,
    )
    return response["JobId"]


def isJobComplete(jobId):
    time.sleep(5)
    client = boto3.client('textract')
    response = client.get_document_analysis(JobId=jobId)
    status = response["JobStatus"]
    print("Job status: {}".format(status))

    while (status == "IN_PROGRESS"):
        time.sleep(5)
        response = client.get_document_analysis(JobId=jobId)
        status = response["JobStatus"]
        print("Job status: {}".format(status))
    return status


def getJobResults(jobId):
    pages = []
    time.sleep(5)
    client = boto3.client('textract')
    response = client.get_document_analysis(JobId=jobId)

    pages.append(response)
    print("Resultset page recieved: {}".format(len(pages)))
    nextToken = None
    if ('NextToken' in response):
        nextToken = response['NextToken']

    while (nextToken):
        time.sleep(5)
        response = client.get_document_analysis(JobId=jobId, NextToken=nextToken)
        pages.append(response)
        print("Resultset page recieved: {}".format(len(pages)))
        nextToken = None
        if 'NextToken' in response:
            nextToken = response['NextToken']
    return pages


# takes a report in S3 and runs textract on it, saving the direct results to disk
def report2textract(fname, write_bucket, features, training=True, report_num=1):
    if '.pdf' in fname:
        docid = fname.rstrip('.pdf')
    else:
        docid = fname
    fname = find_file(docid, report_num=report_num)

    # if report is a .tif: need to download it, convert it to pdf with image2pdf, and upload to s3 again for textract
    # s3: gsq-ml, same object path but pdf
    report_in = '../reports/' + fname
    if '.tif' in fname:
        fname_out = re.sub('.tif', '.pdf', fname)
        report_path = '../reports/' + fname_out
        new_dir = report_path.rsplit('/', 1)[0] + '/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        textloading.download_report(fname, report_in)
        with open(report_path, "wb") as f:
            f.write(img2pdf.convert(open(report_in, "rb")))
        s3 = boto3.resource('s3')
        write_bucket = textsettings.write_bucket
        s3.meta.client.upload_file(report_path, write_bucket, fname_out)
        fname = fname_out
    else:
        new_dir = report_in.rsplit('/', 1)[0] + '/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        textloading.download_report(fname, report_in)

    jobId = startJob(write_bucket, fname, features=features)
    print("Started job with id: {}".format(jobId))
    if isJobComplete(jobId):
        response = getJobResults(jobId)
        if response[0]['JobStatus'] == 'FAILED':
            print(docid + ' failed, status message: ', response[0]['StatusMessage'])
            raise FileNotFoundError
        else:
            #json_response = {'response': response}
            with open(settings.get_full_json_file(docid, training=training, report_num=report_num), 'w') as fp:
                time.sleep(3)  # sometimes the json doesn't get dumped fully, try to stop that
                json.dump(response, fp)
            res_blocks = []
            for i in response:
                res_blocks.extend(i['Blocks'])
            if 'TABLES' in features:
                texttransforming.save_tables(res_blocks, docid, training=training, report_num=report_num)
            if 'FORMS' in features:
                texttransforming.save_kv_pairs(res_blocks, docid, training=training, report_num=report_num)
            print('Completed textracting ' + docid + '_' + str(report_num))
