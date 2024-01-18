# from google.cloud import secretmanager
import json

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
