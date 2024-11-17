# app/main.py
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from app.routers import  data_downloader_router

app = FastAPI(
    title="Bulk Image Vectorizer Service",
    description="Creates Dataset, Vectors and send it to the Database Service",
    version="1.0.0",
)
# Add SessionMiddleware with a secret key
app.add_middleware(SessionMiddleware, secret_key="your_secret_key_here")

# Include Routers
#app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(data_downloader_router.router, prefix="/data", tags=["data"])
#app.include_router(vector_process_router.router, prefix="/vectors", tags=["vectors"])
