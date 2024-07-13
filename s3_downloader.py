import os
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def upload_folder_to_s3(local_folder, bucket_name, s3_folder):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )
    for subdir, dirs, files in os.walk(local_folder):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                s3.upload_fileobj(data, bucket_name, os.path.join(s3_folder, file))
                print(f'Uploaded {file} to s3://{bucket_name}/{s3_folder}/{file}')

def download_folder_from_s3(local_folder, bucket_name, s3_folder):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page['Contents']:
            target_path = os.path.join(local_folder, os.path.relpath(obj['Key'], s3_folder))
            if not os.path.exists(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path))
            s3.download_file(bucket_name, obj['Key'], target_path)
            print(f'Downloaded {obj["Key"]} to {target_path}')

def download_file_from_s3(local_path, bucket_name, s3_key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )
    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))
    s3.download_file(bucket_name, s3_key, local_path)
    print(f'Downloaded {s3_key} to {local_path}')

s3_bucket = 'verizon-audio'
region_name = 'us-east-1'

# download_folder_from_s3('Models/Youtube_en', s3_bucket, 'ANA_en')
# numbers = [164, 169, 174, 179, 184, 189, 194, 199]
# numbers = [79, 84, 89, 94, 99, 104, 109, 114, 119, 124, 129, 134, 139, 144, 149]


# for i in numbers:
#     if len(str(i)) == 2:
#         download_file_from_s3(f'Models/Chris-v4/epoch_2nd_000{i}.pth', s3_bucket, f'models/Chris-V4/epoch_2nd_000{i}.pth')
#     else:
#         download_file_from_s3(f'Models/Chris-v4/epoch_2nd_00{i}.pth', s3_bucket, f'models/Chris-V4/epoch_2nd_00{i}.pth')
#     download_file_from_s3(f'Models/AnaCastano-v1/config_ft.yml', s3_bucket, f'LJSpeech_latest/config_ft.yml')
# upload_folder_to_s3("Models/clarrisa-V1",s3_bucket,"models/clarrisa-V1")
# 'Models/ClarrisaUn', s3_bucket, 'models/ClarrisaUn-V1'
download_file_from_s3('Models/ClarrisaUn-V1/epoch_2nd_00149.pth', s3_bucket, 'models/ClarrisaUn-V1/epoch_2nd_00149.pth')
download_file_from_s3('Models/ClarrisaUn-V1/config_ft.yml', s3_bucket, 'models/ClarrisaUn-V1/config_ft.yml')
# # download_file_from_s3('Models/Rishi1000/epoch_2nd_00089.pth', s3_bucket, 'models/epoch_2nd_00089.pth')
