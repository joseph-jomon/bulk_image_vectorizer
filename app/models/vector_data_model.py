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
