import boto3
import time
import json
import textshowing


def startJob(s3BucketName, objectName, features=None):
    response = None
    client = boto3.client('textract')
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


if __name__ == "__main__":
    s3BucketName = 'gsq-ml'
    pre = 'cr_' # 'smaller_'
    doc_path = '107039' #'89100' #'100697'
    docid = pre + doc_path
    documentName = pre + doc_path + '_1.pdf' #'.pdf'
    features=['TABLES', 'FORMS']
    jobId = startJob(s3BucketName, documentName, features=features)
    print("Started job with id: {}".format(jobId))
    if (isJobComplete(jobId)):
        response = getJobResults(jobId)
        fp = open(docid + '.json', 'w')
        json.dump(response, fp)
        # instead of sending page info individually, concatenate data across them because ids may cross reference and cause errors?
        all_blocks = []
        for pages in response:
            all_blocks.extend(pages['Blocks'])
        short_res = {'Blocks': all_blocks}
        textshowing.save_lines(short_res, outfile=docid+"_text.txt", mode='w')
        if 'TABLES' in features:
            textshowing.save_tables(short_res, outfile=docid+"_tables.csv", mode='w')
        if 'FORMS' in features:
            textshowing.save_kv_pairs(short_res, outfile=docid+"_kvs.csv", mode='w')

        for item in all_blocks:
            if item["BlockType"] == "LINE":
                print('\033[94m' + item["Text"] + '\033[0m')