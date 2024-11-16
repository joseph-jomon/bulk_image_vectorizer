# app/services/data_downloader.py
import os
import re  # Import re module for regular expressions
import pandas as pd
from pathlib import Path
import requests
from fastapi import UploadFile, HTTPException
from app.services.vector_processor import generate_vectors_after_download
output_folder = "./Dataset/images_full"
image_fields = ["mainImage", "onlineImage"]

def sanitize_company_name(company_name: str) -> str:
    sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', company_name)
    return sanitized_name.strip().replace(" ", "_").lower()

def fetch_company_info(cognito_token: str) -> dict:
    """Fetch company info based on the provided cognito token."""
    url = "https://api.production.cloudios.flowfact-prod.cloud/company-service/company"
    headers = {
        'cognitoToken': cognito_token
    }
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

def process_entity(entity_id, cognito_token, company_name):
    entity_endpoint = "https://api.production.cloudios.flowfact-prod.cloud/entity-service/entities"
    headers = {'cognitoToken': cognito_token}
    
    response = requests.get(f'{entity_endpoint}/{entity_id}', headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve data for ID {entity_id}. Status code: {response.status_code}")
        return
    
    entity_data = response.json()
    company_folder = Path(output_folder) / company_name
    entity_folder = company_folder / entity_id
    entity_folder.mkdir(parents=True, exist_ok=True)

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

async def process_csv_and_download_images(file: UploadFile, cognito_token: str):
    """Process CSV and download images."""
    company_info = fetch_company_info(cognito_token)
    company_name = company_info['company_name']
    
    # Read CSV and process each ID
    df = pd.read_csv(file.file)
    if 'id' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain an 'id' column")

    ids = df['id'].tolist()
    for id_value in ids:
        process_entity(id_value, cognito_token, company_name)

    # Trigger vector generation or other post-processing as needed
    await generate_vectors_after_download()
