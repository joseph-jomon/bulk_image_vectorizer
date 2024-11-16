# app/routers/auth_router.py
"""from fastapi import APIRouter, Request, Depends, HTTPException
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
        raise HTTPException(status_code=400, detail=str(e))"""
