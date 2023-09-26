FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN mkdir /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl libgl1 && \
    curl https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/bin/mc && \
    chmod +x /usr/bin/mc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

COPY . .

ENV TORCHDYNAMO_VERBOSE 1
