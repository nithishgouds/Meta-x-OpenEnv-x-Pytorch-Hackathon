FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir .

ENV PORT=7860
EXPOSE 7860

CMD ["python", "server/app.py"]
