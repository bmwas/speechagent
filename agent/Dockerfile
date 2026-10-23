FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./
COPY assets ./assets

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
EXPOSE 8860
CMD ["python", "agent.py"]
