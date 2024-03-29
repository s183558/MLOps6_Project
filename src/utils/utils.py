import json
import os 
from google.cloud import storage
import logging
logger=logging.getLogger(__name__)

def access_secret_version():
    """
    Access the secret version and return the payload.
    """
    # This works well locally, but not in Docker.
    # client = secretmanager.SecretManagerServiceClient()
    # name = "projects/103212519156/secrets/WANDB_API_KEY/versions/latest"
    # response = client.access_secret_version(request={"name": name})
    # return response.payload.data.decode("UTF-8")

    # Opening JSON file
    f = open('WandB_API_key.json')
    
    # returns JSON object as a dictionary
    data = json.load(f)
    
    # Iterating through the json list
    key = data['API_key']
    
    # Closing file
    f.close()

    return key

def get_project_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_bucket(folder: str, bucket_name ="mlops6_tweets"):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    project_dir = get_project_dir()
    logger.info(f'Loading {folder} from {bucket_name}')
    for blob in blobs:
        local_path = os.path.join(project_dir, blob.name)
        print(local_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
