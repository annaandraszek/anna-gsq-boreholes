import boto3
import os


def download_dir(client, resource, dist, local='/tmp', bucket='your_bucket'):
    paginator = client.get_paginator('list_objects')
    reports = ['25843', '25814', '26255', '26446'] # 6mth
            #['64145', '62591', '62764', '57883', '57954', '57956'] # more annual
            #['45644', '69624', '76007'] #hfacr
            #['83636', '83636'] #other
            #['57520', '17706', '22067', '30946', '30941', '77193'] #welcom
            #['127', '103323', '89031', '69056'] # annual
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


if __name__ == "__main__":
    client = boto3.client('s3', region_name='ap-southeast-2')
    resource = boto3.resource('s3')
    download_dir(client, resource, 'QDEX/', '/tmp', bucket='gsq-staging')
