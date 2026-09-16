FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/ data/
COPY frontend/ frontend/
COPY scripts/ scripts/

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Build the vector store at image build time so it's ready on container start.
RUN python scripts/ingest.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
