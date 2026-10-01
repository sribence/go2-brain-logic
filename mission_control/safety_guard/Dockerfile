FROM python:3.11-slim
WORKDIR /app
COPY core/ /app/core/
COPY safety_guard/requirements.txt /app/safety_guard/requirements.txt
RUN pip install --no-cache-dir -r /app/core/requirements.txt -r /app/safety_guard/requirements.txt
COPY safety_guard/ /app/safety_guard/
ENV PYTHONPATH=/app/core:/app/safety_guard
WORKDIR /app/safety_guard
EXPOSE 9115
CMD ["python", "app.py"]
