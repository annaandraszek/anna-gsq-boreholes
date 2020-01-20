
# Running textract on documents in S3

import boto3
import time
import json
import textmain
import texttransforming
import settings
import re
import textloading
import img2pdf
import os


def find_file(docid):  # for finding the file type of a report
    file_pre = settings.get_s3_location(docid, format=None)
    client = boto3.client('s3')
    #my_bucket = s3.Bucket('gsq-staging/QDEX')
    files = client.list_objects_v2(Bucket='gsq-staging', Prefix=file_pre)
    for file in files['Contents']:
        if file['Key'].startswith(file_pre + '.'):
            return file['Key']


def startJob(s3BucketName, objectName, features=None):
    response = None
    client = boto3.client('textract', region_name='ap-southeast-2')
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
        if ('NextToken' in response):
            nextToken = response['NextToken']
    return pages


# takes a report in S3 and runs textract on it, saving the direct results to disk
def report2textract(fname, bucket, features):
    if '.pdf' in fname:
        docid = fname.rstrip('.pdf')
    else:
        docid = fname
    fname = find_file(docid)

    # if report is a .tif: need to download it, convert it to pdf with image2pdf, and upload to s3 again for textract
    # s3: gsq-ml, same object path but pdf
    report_in = 'reports/' + fname
    if '.tif' in fname:
        fname_out = re.sub('.tif', '.pdf', fname)
        report_path = 'reports/' + fname_out
        new_dir = report_path.rsplit('/', 1)[0] + '/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        textloading.download_report(fname, report_in)
        with open(report_path, "wb") as f:
            f.write(img2pdf.convert(open(report_in, "rb")))
        s3 = boto3.resource('s3')
        bucket = 'gsq-ml2'
        s3.meta.client.upload_file(report_path, bucket, fname_out)
        fname = fname_out
    else:
        textloading.download_report(fname, report_in)

    jobId = startJob(bucket, fname, features=features)
    print("Started job with id: {}".format(jobId))
    if isJobComplete(jobId):
        response = getJobResults(jobId)
        if response[0]['JobStatus'] == 'FAILED':
            print(docid + ' failed, status message: ', response[0]['StatusMessage'])
        else:
            fp = open(settings.get_full_json_file(docid), 'w')
            json.dump(response, fp)
            res_blocks = []
            for i in response:
                res_blocks.extend(i['Blocks'])
            if 'TABLES' in features:
                texttransforming.save_tables(res_blocks, docid)
            if 'FORMS' in features:
                texttransforming.save_kv_pairs(res_blocks, docid)

            print('Completed textracting ' + docid)
