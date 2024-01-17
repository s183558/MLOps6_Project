from google.cloud import secretmanager

def access_secret_version():
    """
    Access the secret version and return the payload.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/103212519156/secrets/WANDB_API_KEY/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
