
# Pre-textract functions

import random
#from textracting import report2textract
import boto3
import glob
import img2pdf
import pandas as pd
import settings
import boto3
import os
import logging
from botocore.exceptions import ClientError


def list2strs(lst):
    return [str(e) for e in lst]


def download_dir(client, resource, dist, reports, local='/tmp', bucket='your_bucket'):
    paginator = client.get_paginator('list_objects')

    for report in reports:
        for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist+report+'/'):
            if result.get('CommonPrefixes') is not None:
                for subdir in result.get('CommonPrefixes'):
                    download_dir(client, resource, subdir.get('Prefix'), local, bucket)
            for file in result.get('Contents', []):
                dest_pathname = os.path.join(local, file.get('Key'))
                if not os.path.exists(os.path.dirname(dest_pathname)):
                    os.makedirs(os.path.dirname(dest_pathname))
                resource.meta.client.download_file(bucket, file.get('Key'), dest_pathname)


def download_reports(reports, local_location='reports/'):
    reports = list2strs(reports)
    client = boto3.client('s3', region_name='ap-southeast-2')
    resource = boto3.resource('s3')
    download_dir(client, resource, 'QDEX/', reports, local_location, bucket='gsq-staging')


def download_report(fname, dest):
    resource = boto3.resource('s3')
    resource.meta.client.download_file('gsq-staging', fname, dest)


def get_reportid_sample(num=50):
    random.seed(19)
    reps = pd.read_excel('../investigations/QDEX_reports_BHP.xlsx')
    rs = random.sample(list(reps.REPNO), num)
    return [str(r) for r in rs]


def doc2data(file_id):
    file = settings.get_report_name(file_id, file_extension=True)
    report2textract(file, bucket='gsq-ml', features=['TABLES', 'FORMS'])


def check_if_obj_exists(client, key, bucket):
    response = client.list_objects_v2(
        Bucket=bucket,
        Prefix=key,
    )
    for obj in response.get('Contents', []):
        if obj['Key'] == key:
            return True
        else:
            return False


if __name__ == "__main__":
    report_ids = get_reportid_sample()
    download_reports(report_ids, settings.report_local_path)

    rfs = glob.glob(settings.report_local_path + '*')
    report_ids = set([r.rsplit('\\')[-1] for r in rfs])
    for report in report_ids:
        doc2data(report)