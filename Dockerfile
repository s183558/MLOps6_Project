FROM python:3.10.12-slim

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir 

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# Copy the service account key
COPY sa_key.json sa_key.json

# Set the environment variable for Google Cloud authentication
ENV GOOGLE_APPLICATION_CREDENTIALS=sa_key.json


# Copy the rest of your application
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY conf/ conf/
COPY tests/ tests/

WORKDIR /

RUN pip install -e .

ENTRYPOINT ["python", "-u", "src/train_model.py"]
