FROM python:3.13.0-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Открываем порт для Flask
EXPOSE 5000

CMD ["python", "__main__.py"]