FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование файлов проекта
COPY requirements.txt .
COPY __main__.py .
COPY config.yaml .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Создание директории для моделей
RUN mkdir -p /model-for-text-embedding

# Открытие порта
EXPOSE 5000

# Команда запуска
CMD ["python", "__main__.py"]
