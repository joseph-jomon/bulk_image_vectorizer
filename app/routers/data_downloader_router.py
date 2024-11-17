# app/routers/data_download_router.py
from fastapi import APIRouter, Depends, Request, UploadFile, File, HTTPException
from app.services.data_downloader import process_csv_and_download_images
from app.core.dependencies import get_token

router = APIRouter()

@router.post("/upload-csv/")
async def upload_csv(
    file: UploadFile = File(...),
    token: str = Depends(get_token)  # Use dependency to retrieve token
):
    """
    Endpoint to Download Process Image Data and send Embeddings to the Database Service for Storage
    """
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    await process_csv_and_download_images(file, token)
    return {"status": "CSV processed, images downloading started"}