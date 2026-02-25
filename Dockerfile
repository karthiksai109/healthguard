FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /data /data/ephemeral /data/backup
ENV DATA_DIR=/data
ENV PORT=8080
EXPOSE 8080
CMD ["python", "main.py"]
