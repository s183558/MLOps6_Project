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
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir --no-deps

ADD . /src

RUN pip install -e /src

ENTRYPOINT ["python", "-u", "src/train_model.py"]