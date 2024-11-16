# Dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app /app
WORKDIR /

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
