FROM python:3.10.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc libgl1-mesa-dev libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /fastapi_proj

# Copy only the requirements file, to CACHE the pip install step
COPY app/requirements_fastapi.txt .
RUN pip install --no-cache-dir --upgrade -r requirements_fastapi.txt

COPY app/ ./app/
COPY models/ ./models/
COPY src/ ./src/

WORKDIR /fastapi_proj/app

# Run the FastAPI app
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "80"]
    