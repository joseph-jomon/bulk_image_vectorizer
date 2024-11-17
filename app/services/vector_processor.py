# app/services/vector_processor.py

import os
import numpy as np
import torch
import copy
import datasets 
import httpx
from torchvision import transforms
from transformers import AutoFeatureExtractor, CLIPVisionModelWithProjection
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Model and device setup
model_name = "openai/clip-vit-base-patch32"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model.to(device)
vision_model.eval()

# Directory containing images
root_directory = "./Dataset/images_full"
INGEST_ENDPOINT = "http://database.kundalin.com/ingest_img_bulk/"  # Replace with your database endpoint


# Function to load images from folder structure
def load_image_data(root_directory):
    """Load images into a Hugging Face Dataset."""
    data = {"company_name": [], "id": [], "image_path": []}
    
    for company_folder in os.listdir(root_directory):
        company_path = os.path.join(root_directory, company_folder)
        if os.path.isdir(company_path):
            for entity_folder in os.listdir(company_path):
                entity_path = os.path.join(company_path, entity_folder)
                if os.path.isdir(entity_path):
                    for image_file in os.listdir(entity_path):
                        if image_file.endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(entity_path, image_file)
                            data["company_name"].append(company_folder)
                            data["id"].append(entity_folder)
                            data["image_path"].append(image_path)
    return datasets.Dataset.from_dict(data)
def add_tracking_path(example):
    example["tracking_path"] = example["image_path"]
    return example

# Preprocessing images
def preprocess_images(example):
    """Preprocess images into tensors."""
    extracted = extractor(
        images=[e.convert('RGB') for e in example["image_path"]],
        return_tensors="pt"
    )
    return {
        "pixel_values": extracted["pixel_values"].squeeze(),
        "company_name": example["company_name"],
        "id": example["id"],
        "tracking_path":example["tracking_path"]
    }

# Prepare Dataset and DataLoader
def create_dataloader():
    """Prepare a DataLoader for image data."""
    ds = load_image_data(root_directory)
# Apply the function to modify the dataset permanently
    ds = ds.map(add_tracking_path)
    ds = ds.cast_column("image_path", datasets.Image())
    ds2 = copy.deepcopy(ds)
    ds2.set_transform(preprocess_images)
    image_dl = DataLoader(ds2, batch_size=36, shuffle=False,num_workers=0)
    return DataLoader(ds2, batch_size=36, shuffle=False,num_workers=0)

# Embedding generation
def process_and_generate_embeddings(data_loader):
    """Generate embeddings for images."""
    embeddings = []
    metadata = []

    for batch in tqdm(data_loader):
        with torch.no_grad():
            pixel_values = batch["pixel_values"].to(device)
            output = vision_model(pixel_values).image_embeds.squeeze()
            embeddings.append(output.to("cpu"))

        # Append metadata for each image
        for i in range(len(batch["company_name"])):
            metadata.append({
                "company_name": batch["company_name"][i],
                "id": batch["id"][i],
                "tracking_path": batch["tracking_path"][i]
            })
    vision_embeddings = np.vstack(embeddings)
    vision_embeddings_normed = vision_embeddings / np.linalg.norm(vision_embeddings,axis=1)[:, np.newaxis]

    # Store norlaized embeddings 


    return vision_embeddings_normed, metadata

# Batch ingestion
async def send_vectors_in_batches(embeddings, metadata, batch_size=100):
    """Send embeddings in batches to the database."""
    async with httpx.AsyncClient() as client:
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]

            payload = [
                {
                    "company_name": meta["company_name"],
                    "id": meta["id"],
                    "image_embedding": embedding.tolist(),
                    "tracking_path": meta["tracking_path"]
                }
                for embedding, meta in zip(batch_embeddings, batch_metadata)
            ]

            response = await client.post(
                INGEST_ENDPOINT,
                params={"index_name": f"{batch_metadata[i]['company_name']}_bulk_image"}, 
                json={"items": payload})
            if response.status_code != 200:
                print(f"Batch ingestion failed: {response.json()}")

# Orchestrator function
async def generate_vectors_after_download():
    """Orchestrate the entire vector generation process."""
    data_loader = create_dataloader()
    embeddings, metadata = process_and_generate_embeddings(data_loader)
    #print(f"the embedding is {embeddings} the metadata is {metadata}")
    await send_vectors_in_batches(embeddings, metadata)
