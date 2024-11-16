# app/utils/helpers.py
import httpx

async def authenticate_api_key(api_key: str) -> str:
    """Authenticate the API key and return a cognito token."""
    url = "https://api.production.cloudios.flowfact-prod.cloud/admin-token-service/public/adminUser/authenticate"
    headers = {'token': api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Invalid API key")
