FROM python:3.11.6

RUN apt update -y && apt upgrade -y \
    && apt install -y build-essential curl \
    && apt clean

# Install build utilities for Qt, OCR, etc..
RUN apt install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 lzma liblzma-dev libbz2-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev

# Upgrade pip to the latest version
RUN pip install --upgrade pip build wheel

WORKDIR /workspace

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

