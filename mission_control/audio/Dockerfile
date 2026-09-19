FROM python:3.11-slim
RUN apt-get update && apt-get install -y gcc build-essential libasound2-dev alsa-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY core/ /app/core/
COPY audio/requirements.txt /app/audio/requirements.txt
RUN pip install --no-cache-dir -r /app/core/requirements.txt -r /app/audio/requirements.txt
COPY audio/ /app/audio/
ENV PYTHONPATH=/app/core:/app/audio
WORKDIR /app/audio
CMD ["python", "app.py"]
