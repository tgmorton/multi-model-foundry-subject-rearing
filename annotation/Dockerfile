FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn python-multipart

COPY . /app/annotation/

EXPOSE 8643

CMD ["python", "-m", "annotation", "--host", "0.0.0.0", "--port", "8643"]
