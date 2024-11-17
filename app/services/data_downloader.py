# app/services/data_downloader.py
import os
import re  # Import re module for regular expressions
import pandas as pd
import requests
import logging
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from fastapi import UploadFile, HTTPException
from app.services.vector_processor import generate_vectors_after_download
output_folder = "./Dataset/images_full"
image_fields = ["mainImage", "onlineImage"]

# Configure logging
logging.basicConfig(
    filename="image_download_error.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def log_error(message):
    logging.error(message)

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
    """Download an image and verify its validity."""
    try:
        response = requests.get(uri, stream=True, timeout=10)  # Add a timeout for unresponsive URLs
        if response.status_code == 200:
            # Save the file
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {save_path}")
            # Verify the image file
            try:
                with Image.open(save_path) as img:
                    img.verify()  # Check if the image file is valid
            except UnidentifiedImageError:
                log_error(f"Invalid image file: {save_path}")
                os.remove(save_path)  # Remove the invalid file
        else:
            log_error(f"Failed to download image from {uri}. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        log_error(f"Request error for {uri}: {e}")
    except Exception as e:
        log_error(f"Unexpected error for {uri}: {e}")

def process_entity(entity_id, cognito_token, company_name):
    """Process an entity by downloading its images."""
    try:
        entity_endpoint = "https://api.production.cloudios.flowfact-prod.cloud/entity-service/entities"
        headers = {'cognitoToken': cognito_token}
        
        response = requests.get(f'{entity_endpoint}/{entity_id}', headers=headers)
        if response.status_code != 200:
            log_error(f"Failed to retrieve data for ID {entity_id}. Status code: {response.status_code}")
            return  # Skip this ID

        entity_data = response.json()
        company_folder = Path(output_folder) / company_name
        entity_folder = company_folder / entity_id
        entity_folder.mkdir(parents=True, exist_ok=True)

        # Process image fields
        for field in image_fields:
            if field in entity_data and entity_data[field].get("values"):
                for idx, image_info in enumerate(entity_data[field]["values"]):
                    image_uri = image_info.get("uri")
                    if not image_uri or not isinstance(image_uri, str):  # Check for valid URI
                        log_error(f"Invalid or missing URI in field {field} for ID {entity_id}")
                        continue  # Skip this image
                    image_filename = f"{field}_image_{idx + 1}.jpg"
                    image_path = entity_folder / image_filename
                    download_image(image_uri, image_path)  # Call the download function
    except Exception as e:
        log_error(f"Error processing entity ID {entity_id}: {e}")

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
