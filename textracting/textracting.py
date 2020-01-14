
# Running textract on documents in S3

import boto3
import time
import json
import textmain
import texttransforming
import settings


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
    try:
        fname = settings.get_s3_location(docid)
        jobId = startJob(bucket, fname, features=features)
    except:
        fname = settings.get_s3_location(docid, format='tif')
        jobId = startJob(bucket, fname, features=features)
    print("Started job with id: {}".format(jobId))
    if isJobComplete(jobId):
        response = getJobResults(jobId)
        if response[0]['JobStatus'] == 'FAILED':
            print(docid + ' failed, status message: ', response[0]['StatusMessage'])
        else:
            fp = open(settings.get_full_json_file(docid), 'w')
            json.dump(response, fp)

            if 'TABLES' in features:
                textmain.save_tables(response, docid)
            if 'FORMS' in features:
                textmain.save_kv_pairs(response, docid)

            print('Completed textracting ' + docid)
