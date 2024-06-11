import os
import requests
import zipfile

# Define directory paths
DATASETS_DIR = "/teamspace/studios/this_studio/vqa_implementation/datasets"
ANNOTATIONS_DIR = os.path.join(DATASETS_DIR, "Annotations")
QUESTIONS_DIR = os.path.join(DATASETS_DIR, "Questions")
IMAGES_DIR = os.path.join(DATASETS_DIR, "Images")

# Create directories
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(QUESTIONS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Function to download a file
def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    print(f"Downloaded {dest}.")

# Function to extract a zip file
def extract_zip(file_path, dest_dir):
    print(f"Extracting {file_path} to {dest_dir}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f"Extracted {file_path}.")

# Function to remove a file
def remove_file(file_path):
    print(f"Removing {file_path}...")
    os.remove(file_path)
    print(f"Removed {file_path}.")

# URLs for the datasets
datasets = {
    "annotations": [
        ("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip", os.path.join(ANNOTATIONS_DIR, "v2_Annotations_Train_mscoco.zip")),
        ("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip", os.path.join(ANNOTATIONS_DIR, "v2_Annotations_Val_mscoco.zip"))
    ],
    "questions": [
        ("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip", os.path.join(QUESTIONS_DIR, "v2_Questions_Train_mscoco.zip")),
        ("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip", os.path.join(QUESTIONS_DIR, "v2_Questions_Val_mscoco.zip")),
        ("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip", os.path.join(QUESTIONS_DIR, "v2_Questions_Test_mscoco.zip"))
    ],
    "images": [
        ("http://images.cocodataset.org/zips/train2014.zip", os.path.join(IMAGES_DIR, "train2014.zip")),
        ("http://images.cocodataset.org/zips/val2014.zip", os.path.join(IMAGES_DIR, "val2014.zip")),
        ("http://images.cocodataset.org/zips/test2015.zip", os.path.join(IMAGES_DIR, "test2015.zip"))
    ]
}

# Download, extract, and remove zip files for annotations
for url, path in datasets["annotations"]:
    download_file(url, path)
    extract_zip(path, ANNOTATIONS_DIR)
    remove_file(path)

# Download, extract, and remove zip files for questions
for url, path in datasets["questions"]:
    download_file(url, path)
    extract_zip(path, QUESTIONS_DIR)
    remove_file(path)

# Download, extract, and remove zip files for images
for url, path in datasets["images"]:
    download_file(url, path)
    extract_zip(path, IMAGES_DIR)
    remove_file(path)
