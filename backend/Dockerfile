# Dockerfile

FROM python:3.9

WORKDIR /app

COPY requirements.txt .

# Install dependencies including the missing one
RUN pip install --no-cache-dir -r requirements.txt importlib_metadata

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
