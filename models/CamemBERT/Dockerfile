FROM python:3.10-slim

WORKDIR /home/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && \
    chown -R appuser:appuser /home/app

USER appuser

ENTRYPOINT ["python", "train.py"]
