import boto3
import os


def list2strs(lst):
    return [str(e) for e in lst]


def download_dir(client, resource, dist, reports, local='/tmp', bucket='your_bucket', ):
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


def download_reports(reports, local_location='tmp/'):
    reports = list2strs(reports)
    client = boto3.client('s3', region_name='ap-southeast-2')
    resource = boto3.resource('s3')
    download_dir(client, resource, 'QDEX/', reports, local_location, bucket='gsq-staging')


if __name__ == "__main__":
   reports = [2394, 458, 32422]
   download_reports(reports)



