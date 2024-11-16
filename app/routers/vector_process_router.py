"""# app/routers/vector_process_router.py
from fastapi import APIRouter, Depends, HTTPException
from app.services.vector_processor import generate_and_store_vectors
from app.models.vector_data_model import VectorDataBatch

router = APIRouter()

@router.post("/process-vectors/")
async def process_vectors(batch: VectorDataBatch):
    try:
        await generate_and_store_vectors(batch)
        return {"status": "vectors processed and stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process and store vectors: {str(e)}")"""
