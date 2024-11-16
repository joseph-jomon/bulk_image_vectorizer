# app/core/dependencies.py
from fastapi import Depends, HTTPException, Request
from app.utils.helpers import authenticate_api_key

async def get_token(request: Request, api_key: str):
    """
    Dependency function to authenticate API key and retrieve a token.
    If already authenticated, retrieves the token from the session.
    """
    token = ''
    #request.session.get("token")
    if not token:
        # Authenticate and store token in session
        try:
            token = await authenticate_api_key(api_key)
            request.session["token"] = token
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid API key")

    return token
