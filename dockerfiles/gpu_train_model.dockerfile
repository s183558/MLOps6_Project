# Base image
FROM python:3.10.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY conf/ conf/
COPY tests/ tests/

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN pip install -e .

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc *.dvc

RUN gcloud auth activate-service-account ${{ secrets.GDRIVE_CREDENTIALS_DATA }}

RUN dvc config core.no_scm true
RUN dvc pull

ENTRYPOINT ["python", "-u", "src/entry.py"]