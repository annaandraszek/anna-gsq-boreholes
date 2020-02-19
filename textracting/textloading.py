## @file
# Pre-textract functions

import random
import pandas as pd
import boto3
import os
from datetime import timedelta, date


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
    download_dir(client, resource, 'QDEX/', reports, local_location, bucket='gsq-horizon')


def download_report(fname, dest):
    s3 = boto3.resource('s3')
    s3.Bucket('gsq-horizon').download_file(fname, dest)


def col2datetime(df, col):  # to stop to_datetime from converting 1900s to 2000s
    df[col] = pd.to_datetime(df[col], dayfirst=True)  # convert col to datetime - will convert all to 2000s
    future = df[col] > pd.Timestamp(year=2020,month=1,day=1)  # find dates that are in the future and therefore wrong
    df.loc[future, col] -= timedelta(days=365.25*100)  # change future dates to be 100 years in the past; 2089 -> 1989
    return df


def edit_reports_xlsx():
    reps = pd.read_excel('../investigations/QDEX_metada_export.xlsx')  # _reports_BHP.xlsx')
    reps = col2datetime(reps, 'REPDATE')
    reps = col2datetime(reps, 'RECDATE')
    chrono = reps['REPDATE'] < reps['RECDATE']    # check if repdate is after recdate
    reps.loc[chrono, 'REPDATE'] -= timedelta(days=365.25*100)  # subtract 100 years from the dates that were found to fall after the RECDATE
    reps.to_excel('../investigations/QDEX_export_v2.xlsx')
    return reps


def get_reportid_sample(num=50, submitter=None, rtype_exclude=None, cutoffdate=pd.Timestamp(1990, 1, 1)):
    #random.seed(19)
    #reps = edit_reports_xlsx()
    reps = pd.read_excel('../investigations/QDEX_export_v2.xlsx')#_reports_BHP.xlsx')
    reps = reps.loc[reps.RSTATUS.str.contains('C') == False]
    reps.REPNO = reps.REPNO.astype(int)
    #cutoffdate = pd.Timestamp(1990, 1, 1)
    #time_mask = reps.REPDATE > cutoffdate

    if cutoffdate:
        reps = reps.loc[reps.REPDATE > cutoffdate]
    if submitter:
        reps = reps.loc[reps.SUBMITBY.str.contains(submitter) == True]
    if rtype_exclude:
        reps = reps.loc[reps.RTYPE.str.contains(rtype_exclude) == False]
    rs = random.sample(list(reps.REPNO), num)
    return [str(r) for r in rs]



# def doc2data(file_id):
#     file = settings.get_report_name(file_id, file_extension=True)
#     report2textract(file, bucket='gsq-ml', features=['TABLES', 'FORMS'])


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


# if __name__ == "__main__":
#     report_ids = get_reportid_sample()
#     download_reports(report_ids, settings.report_local_path)
#
#     rfs = glob.glob(settings.report_local_path + '*')
#     report_ids = set([r.rsplit('\\')[-1] for r in rfs])
#     for report in report_ids:
#         doc2data(report)