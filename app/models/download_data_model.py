# app/models/download_data_model.py
from pydantic import BaseModel
from typing import List
from fastapi import UploadFile

class CSVFileUpload(BaseModel):
    file: UploadFile
