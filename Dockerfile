FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv/self-lora

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service.py .
COPY trainers ./trainers
COPY scripts ./scripts

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8010"]
