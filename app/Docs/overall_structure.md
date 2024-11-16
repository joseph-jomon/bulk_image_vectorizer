To build a scalable, maintainable FastAPI application that incorporates image data downloading, vector generation, and batch submission to an Elasticsearch database, we'll create a structured project adhering to SOLID principles. I’ll provide the folder structure, code for each file, Docker configuration for both development and production, and explanations on how each part aligns with best practices.

### Folder Structure

```plaintext
project/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                 # Settings and configurations
│   │   ├── dependencies.py           # Dependency injection functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vector_data_model.py      # Pydantic models for API data validation
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth_router.py            # Authentication endpoints
│   │   ├── data_download_router.py   # Image download and dataset creation
│   │   ├── vector_process_router.py  # Vector creation and database ingestion
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_downloader.py        # Image download and folder management
│   │   ├── vector_processor.py       # Vector generation and batching
│   │   ├── db_client.py              # Database client for Elasticsearch
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py                # General helper functions
│
├── Dockerfile                        # Dockerfile for production
├── docker-compose.yml                # Docker Compose for production
├── docker-compose.debug.yml          # Docker Compose for debug
└── requirements.txt                  # Python package requirements
```

---

### Code for Each File

#### `app/core/config.py`

Configuration settings are centralized here. Use environment variables for secrets.

```python
# app/core/config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    BASE_URL: str = "https://api.production.cloudios.flowfact-prod.cloud"
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
    API_KEY: str = os.getenv("API_KEY", "your_default_api_key")
    TIMEOUT: int = int(os.getenv("TIMEOUT", "30"))

settings = Settings()
```

#### `app/core/dependencies.py`

Dependency functions for FastAPI routes.

```python
# app/core/dependencies.py
from fastapi import Depends, Request, HTTPException
from app.services.db_client import get_db_client
from app.core.config import settings
from app.utils.helpers import authenticate_api_key

async def get_token(request: Request):
    token = request.session.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token

def get_database_client():
    return get_db_client(settings.ELASTICSEARCH_HOST)
```

#### `app/models/vector_data_model.py`

Pydantic models to validate incoming data.

```python
# app/models/vector_data_model.py
from pydantic import BaseModel
from typing import List

class VectorDataItem(BaseModel):
    id: str
    company_name: str
    vector_embedding: List[float]
    tracking_path: str

class VectorDataBatch(BaseModel):
    items: List[VectorDataItem]
```

#### `app/routers/auth_router.py`

Authentication endpoints to handle API key and session setup.

```python
# app/routers/auth_router.py
from fastapi import APIRouter, Request, Depends, HTTPException
from app.core.config import settings
from app.utils.helpers import authenticate_api_key

router = APIRouter()

@router.get("/auth/")
async def auth_api_key(api_key: str, request: Request):
    try:
        token = await authenticate_api_key(api_key)
        request.session["token"] = token
        return {"status": "success", "token": token}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

#### `app/routers/data_download_router.py`

Endpoint to download images and organize them in folders.

```python
# app/routers/data_download_router.py
from fastapi import APIRouter, Depends, Request
from app.services.data_downloader import download_and_create_dataset
from app.core.dependencies import get_token

router = APIRouter()

@router.post("/download-data/")
async def download_data(request: Request, token: str = Depends(get_token)):
    company_name = request.session.get("company_name")
    download_and_create_dataset(company_name, token)
    return {"status": "dataset created"}
```

#### `app/routers/vector_process_router.py`

Endpoint for vector generation and ingestion into the database.

```python
# app/routers/vector_process_router.py
from fastapi import APIRouter, Depends
from app.services.vector_processor import generate_and_store_vectors
from app.core.dependencies import get_database_client
from app.models.vector_data_model import VectorDataBatch

router = APIRouter()

@router.post("/process-vectors/")
async def process_vectors(batch: VectorDataBatch, db_client=Depends(get_database_client)):
    await generate_and_store_vectors(batch, db_client)
    return {"status": "vectors processed and stored"}
```

#### `app/services/data_downloader.py`

Module to handle downloading images and saving them in a structured directory.

```python
# app/services/data_downloader.py
import os
from pathlib import Path
import requests

def download_and_create_dataset(company_name, token):
    # Directory creation and image download logic here
    pass
```

#### `app/services/vector_processor.py`

Module to process images and create vectors, then send in batches to the database.

```python
# app/services/vector_processor.py
import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection, AutoFeatureExtractor
from app.models.vector_data_model import VectorDataBatch

def generate_and_store_vectors(batch: VectorDataBatch, db_client):
    # Vector generation and storing logic here
    pass
```

#### `app/services/db_client.py`

Module to manage Elasticsearch client and operations.

```python
# app/services/db_client.py
from elasticsearch import AsyncElasticsearch

def get_db_client(host):
    return AsyncElasticsearch(hosts=[host])
```

#### `app/utils/helpers.py`

Helper functions like API key authentication and company name sanitization.

```python
# app/utils/helpers.py
import httpx
import re

def sanitize_company_name(company_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '', company_name).replace(" ", "_").lower()

async def authenticate_api_key(api_key: str):
    url = "https://api.production.cloudios.flowfact-prod.cloud/admin-token-service/public/adminUser/authenticate"
    headers = {"token": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    return response.text
```

---

### `main.py`

Initialize the FastAPI app and include routers.

```python
# app/main.py
from fastapi import FastAPI
from app.routers import auth_router, data_download_router, vector_process_router

app = FastAPI()

# Include Routers
app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(data_download_router.router, prefix="/data", tags=["data"])
app.include_router(vector_process_router.router, prefix="/vectors", tags=["vectors"])
```

### `Dockerfile`

Dockerfile for production setup.

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app /app
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

### `docker-compose.yml` for Production

```yaml
version: '3.8'
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      ELASTICSEARCH_HOST: http://elasticsearch:9200
    depends_on:
      - elasticsearch
    ports:
      - "8000:80"

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.10.1
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"
```

### `docker-compose.debug.yml` for Development

```yaml
version: '3.8'
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      ELASTICSEARCH_HOST: http://elasticsearch:9200
    volumes:
      - ./app:/app
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
    depends_on:
      - elasticsearch
    ports:
      - "8000:80"

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.10.1
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"
```

### `requirements.txt`

Dependencies for the FastAPI app.

```plaintext
fastapi
httpx
uvicorn
pandas
torch
transformers
elasticsearch
```

---

This structure