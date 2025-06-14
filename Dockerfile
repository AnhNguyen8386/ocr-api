FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    git gcc libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*


COPY . /app


RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
