FROM python:3.10

WORKDIR /home/app

RUN apt-get update -y && \
    apt-get install -y nano unzip libgl1 curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

RUN useradd -m appuser

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

COPY . /home/app

USER appuser

CMD gunicorn app:app  --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker 