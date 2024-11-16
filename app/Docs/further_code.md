Given that we’ll use an endpoint to send vector data in batches to Elasticsearch instead of a direct database client, I’ll adjust the `vector_processor.py` and `data_downloader.py` files accordingly. This approach will use HTTP requests to interact with the external ingestion endpoint (`/ingest_img_bulk/`) for storing vectors in Elasticsearch. Below, I’ve provided the updated code for `vector_processor.py` and `data_downloader.py`.

---

### `app/services/data_downloader.py`

This module handles the process of authenticating, fetching entity data, and downloading images into a structured directory based on the company name and object ID.

```python
# app/services/data_downloader.py
import os
import re
from pathlib import Path
import requests

# Define the output folder for saving images
output_folder = "./Dataset/images_full"
image_fields = ["mainImage", "onlineImage"]  # Fields in the data that contain images

def sanitize_company_name(company_name: str) -> str:
    """Sanitize the company name to ensure valid directory names."""
    sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', company_name)
    return sanitized_name.strip().replace(" ", "_").lower()

def authenticate_api_key(api_key: str) -> str:
    """Authenticate with the provided API key and get a cognito token."""
    url = "https://api.production.cloudios.flowfact-prod.cloud/admin-token-service/public/adminUser/authenticate"
    headers = {'token': api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text  # Cognito token
    else:
        raise Exception("Invalid API key")

def fetch_company_info(cognito_token: str) -> dict:
    """Fetch and sanitize company info based on the provided cognito token."""
    url = "https://api.production.cloudios.flowfact-prod.cloud/company-service/company"
    headers = {'cognitoToken': cognito_token}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        company_info = response.json()
        return {
            'company_id': company_info['id'],
            'company_name': sanitize_company_name(company_info['companyName'])
        }
    else:
        raise Exception("Failed to fetch company info")

def download_image(uri, save_path):
    """Download an image from a URI and save it to a specified path."""
    try:
        response = requests.get(uri, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
                print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download image from {uri}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image from {uri}: {e}")

def process_entity(row, cognito_token, company_name):
    """Process each entity to download images into organized folders."""
    entity_id = row['id']
    entity_endpoint = "https://api.production.cloudios.flowfact-prod.cloud/entity-service/entities"
    headers = {'cognitoToken': cognito_token}
    
    # Fetch entity data
    response = requests.get(f'{entity_endpoint}/{entity_id}', headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve data for ID {entity_id}. Status code: {response.status_code}")
        return
    
    entity_data = response.json()
    
    # Create folder paths based on company and entity ID
    company_folder = Path(output_folder) / company_name
    entity_folder = company_folder / entity_id
    entity_folder.mkdir(parents=True, exist_ok=True)

    # Download images from specified fields
    for field in image_fields:
        if field in entity_data and entity_data[field].get("values"):
            for idx, image_info in enumerate(entity_data[field]["values"]):
                image_uri = image_info.get("uri")
                if image_uri:
                    image_filename = f"{field}_image_{idx + 1}.jpg"
                    image_path = entity_folder / image_filename
                    download_image(image_uri, image_path)
                else:
                    print(f"No URI found in field {field} for ID {entity_id}")

def download_and_create_dataset(api_key: str, dataset):
    """Main function to authenticate, fetch company info, and download images."""
    try:
        cognito_token = authenticate_api_key(api_key)
        company_info = fetch_company_info(cognito_token)
        company_name = company_info['company_name']
        
        # Process each row in the dataset to download images
        dataset.apply(lambda row: process_entity(row, cognito_token, company_name), axis=1)
    except Exception as e:
        print(f"Error in download_and_create_dataset: {e}")
```

---

### `app/services/vector_processor.py`

This module processes images, generates vectors, and sends them to the `/ingest_img_bulk/` endpoint in batches.

```python
# app/services/vector_processor.py
import os
import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection, AutoFeatureExtractor
from torch.utils.data import DataLoader
from PIL import Image
import httpx
from tqdm import tqdm

# Set up model and device
model_name = "openai/clip-vit-base-patch32"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model.to(device)
vision_model.eval()

# Define the endpoint for batch ingestion
INGEST_ENDPOINT = "http://database-service-url/ingest_img_bulk/"  # Replace with the actual database service URL

def load_image_data(root_directory):
    """Load images from the root directory into a structured format."""
    data = {"company_name": [], "id": [], "image_path": []}
    for company_folder in os.listdir(root_directory):
        company_path = os.path.join(root_directory, company_folder)
        if os.path.isdir(company_path):
            for entity_id_folder in os.listdir(company_path):
                entity_path = os.path.join(company_path, entity_id_folder)
                if os.path.isdir(entity_path):
                    for image_file in os.listdir(entity_path):
                        if image_file.endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(entity_path, image_file)
                            data["company_name"].append(company_folder)
                            data["id"].append(entity_id_folder)
                            data["image_path"].append(image_path)
    return data

def preprocess_image(image_path):
    """Load and preprocess a single image."""
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    return inputs["pixel_values"]

def generate_embeddings(data, batch_size=36):
    """Generate embeddings from a batch of images."""
    image_data = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings, metadata = [], []
    for batch in tqdm(image_data):
        pixel_values = torch.stack([preprocess_image(img) for img in batch["image_path"]]).to(device)
        with torch.no_grad():
            output = vision_model(pixel_values).image_embeds
            normalized_output = output / output.norm(dim=1, keepdim=True)
            embeddings.extend(normalized_output.cpu().numpy())
        
        metadata.extend([{
            "company_name": batch["company_name"][i],
            "id": batch["id"][i],
            "tracking_path": batch["image_path"][i]
        } for i in range(len(batch["company_name"]))])
    
    return embeddings, metadata

async def send_vectors_in_batches(embeddings, metadata, batch_size=100):
    """Send vectors in batches to the database ingestion endpoint."""
    async with httpx.AsyncClient() as client:
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            batch_payload = [{
                "company_name": meta["company_name"],
                "id": meta["id"],
                "vector_embedding": embedding.tolist(),
                "tracking_path": meta["tracking_path"]
            } for embedding, meta in zip(batch_embeddings, batch_metadata)]
            
            response = await client.post(INGEST_ENDPOINT, json={"items": batch_payload})
            if response.status_code != 200:
                print(f"Batch ingestion failed: {response.json()}")

def process_and_store_embeddings(root_directory):
    """Main function to load data, generate vectors, and send in batches."""
    data = load_image_data(root_directory)
    embeddings, metadata = generate_embeddings(data)
    torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
    return send_vectors_in_batches(embeddings, metadata)
```

---

### `app/main.py`

Set up the main application to include the required routers and run the data ingestion pipeline.

```python
# app/main.py
from fastapi import FastAPI, Depends
from app.routers import auth_router, data_download_router, vector_process_router

app = FastAPI()

app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(data_download_router.router, prefix="/data", tags=["data"])
app.include_router(vector_process_router.router, prefix="/vectors", tags=["vectors"])
```

---

These modules together provide a solid foundation for:
1. Authenticating and obtaining tokens.
2. Downloading images and creating an organized dataset.
3. Generating image embeddings.
4. Sending vectors to the database service in batches using HTTP requests. 

With this setup, you can containerize and deploy the application to integrate seamlessly with an external Elasticsearch-based database service via an ingestion endpoint.