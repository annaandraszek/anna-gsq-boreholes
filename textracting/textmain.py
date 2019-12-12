import boto3
import time
import json
import textshowing
import textracting
import settings

def startJob(s3BucketName, objectName, features=None):
    response = None
    client = boto3.client('textract', region_name='ap-southeast-1')
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



def file2doc(fname, bucket, features, pageinfo=False, ret=False):
    if '.pdf' in fname:
        docid = fname.rstrip('.pdf')
    else:
        docid = fname
    jobId = startJob(bucket, fname, features=features)
    print("Started job with id: {}".format(jobId))
    if isJobComplete(jobId):
        response = getJobResults(jobId)
        fp = open(settings.get_full_json_file(docid), 'w')
        json.dump(response, fp)
        # instead of sending page info individually, concatenate data across them because ids may cross reference and cause errors?
        short_res = textshowing.json2res(response)
        textshowing.save_lines(short_res, docid)
        if pageinfo:
            pginfo = textshowing.save_pageinfo(short_res, docid)
            pglines = textshowing.save_pagelines(short_res,docid)
        if 'TABLES' in features:
            textshowing.save_tables(short_res, docid)
        if 'FORMS' in features:
            textshowing.save_kv_pairs(short_res, docid)

        # for item in all_blocks:
        #     if item["BlockType"] == "LINE":
        #         print('\033[94m' + item["Text"] + '\033[0m')
        print('Completed ' + docid)
        if pageinfo and ret:
            return pginfo, pglines


if __name__ == "__main__":
    s3BucketName = 'gsq-ml'
    pre = 'cr_' # 'smaller_'
    doc_path = '103323' #'89100' #'100697'
    docid = pre + doc_path
    documentName = pre + doc_path + '_1.pdf' #'.pdf'
    features=['TABLES', 'FORMS']
    file2doc(documentName, s3BucketName, features)