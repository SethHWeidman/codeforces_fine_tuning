import getpass
import os
from os import path

import boto3
from botocore import exceptions

# —————— CONFIG ——————
S3_BUCKET_NAME = "codeforces-fine-tuning"
S3_CLIENT = boto3.client("s3")  # make sure your AWS creds & region are configured

AWS_ACCESS_KEY_ID = getpass.getpass("Enter your AWS_ACCESS_KEY_ID: ")
AWS_SECRET_ACCESS_KEY = getpass.getpass("Enter your AWS_SECRET_ACCESS_KEY: ")

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY


# —————— BUCKET SETUP ——————
def create_bucket_if_needed(bucket_name: str, region: str | None = None) -> None:
    """
    Ensure S3 bucket exists. If the bucket is already owned by you,
    do nothing. Otherwise create it in the given region.
    """
    try:
        # Cheap HEAD request – succeeds (200) if bucket exists & you own it
        S3_CLIENT.head_bucket(Bucket=bucket_name)
        print(f"Bucket already exists: {bucket_name}")
        return
    except exceptions.ClientError as e:
        code = int(e.response["Error"]["Code"])
        if code != 404:
            # Some other error (e.g. you *don't* own the bucket) → re-raise
            raise

    # Bucket not found – create it
    if region and region != "us-east-1":
        S3_CLIENT.create_bucket(
            Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
        )
    else:  # us-east-1 uses legacy call with no configuration block
        S3_CLIENT.create_bucket(Bucket=bucket_name)

    print(f"Bucket created: {bucket_name}")


def upload_folder_to_s3(
    local_folder: str, s3_prefix: str, bucket_name: str = S3_BUCKET_NAME
) -> None:
    """
    Walks local_folder, and for each file uploads it to
    s3://{bucket_name}/{s3_prefix}/{relative_path}.
    """
    for root, _, files in os.walk(local_folder):
        for fname in files:
            local_path = path.join(root, fname)
            # build the key so subdirectories are preserved
            rel_path = path.relpath(local_path, local_folder)
            s3_key = f"{s3_prefix}/{rel_path}"
            print(f"Uploading {local_path} → s3://{bucket_name}/{s3_key}")
            S3_CLIENT.upload_file(Filename=local_path, Bucket=bucket_name, Key=s3_key)


if __name__ == "__main__":

    create_bucket_if_needed(S3_BUCKET_NAME)
    for problem_level in [900, 1000, 1100]:

        upload_folder_to_s3(
            f'statements/{problem_level}',
            f'statements/{problem_level}',
            bucket_name=S3_BUCKET_NAME,
        )
        upload_folder_to_s3(
            f'tests_verified/{problem_level}',
            f'tests_verified/{problem_level}',
            bucket_name=S3_BUCKET_NAME,
        )
