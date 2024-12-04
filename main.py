import zipfile
import json
import boto3
import os
from dotenv import load_dotenv
import boto3
import os

load_dotenv()

sns = boto3.client("sns", region_name="us-east-1")
s3 = boto3.client("s3")
ec2 = boto3.client("ec2")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
SNS_ARN = os.getenv("SNS_ARN")
DEPLOYMENT = os.getenv("DEPLOYMENT")


def download_dataset_from_s3(name):
    s3_key = os.path.join("dataset", name)
    s3.download_file(S3_BUCKET_NAME, s3_key, name)


def train_yolo():
    os.makedirs("runs", exist_ok=True)
    with open("runs/tmp.txt", "w") as f:
        f.write("Hello S3")
    print("trained")


def upload_to_s3(local_path, s3_path, zip_name="upload.zip"):
    zip_path = os.path.join("/tmp", zip_name)  # Temporary path for the zip file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, local_path))

    if DEPLOYMENT == "dev":
        print("Not uploading in dev env")
        return
    s3_key = os.path.join(s3_path, zip_name)

    s3.upload_file(zip_path, S3_BUCKET_NAME, s3_key)
    print(f"Uploaded {zip_path} to s3://{S3_BUCKET_NAME}/{s3_key}")


def send_done_signal():
    # try:
    sns.publish(
        TargetArn=SNS_ARN,
        Message="Done training",
        Subject="Testing",
    )


# except Exception as e:
#     print("Failed to send message")
#     pass


if __name__ == "__main__":
    if DEPLOYMENT != "dev":
        download_dataset_from_s3("yolo.zip")
    train_yolo()
    upload_to_s3("./runs", "yolo_runs/", "yolo_runs.zip")
    send_done_signal()
