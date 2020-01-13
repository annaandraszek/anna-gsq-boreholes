
# Running textract on documents in S3

import boto3
import time
import json
import textmain
import texttransforming
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


# takes a report in S3 and runs textract on it, saving the direct results to disk
def report2textract(fname, bucket, features):
    if '.pdf' in fname:
        docid = fname.rstrip('.pdf')
    else:
        docid = fname
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


if __name__ == "__main__":
    s3BucketName = 'gsq-ml'
    pre = 'cr_' # 'smaller_'
    docs = ['30281', '31069', '33412', '37414', '37838', '38865', '44387', '45470', '47884', '56500', '57048']
    for doc_path in docs:
        docid = pre + doc_path
        documentName = pre + doc_path + '_1.pdf' #'.pdf'
        features=['TABLES', 'FORMS']
        report2textract(documentName, s3BucketName, features)
        textmain.clean_and_restruct(docid) # pagelineinfo -> cleanpage -> restructpageinfo
