# Stage 1: Build and train the model
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies and build essentials if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

# Train and persist the model during build to speed up container start
RUN python -c "import os, pandas as pd; from recommender import HybridRecommender; BASE='.'; customers=pd.read_csv(os.path.join(BASE,'data','customers.csv')); interactions=pd.read_csv(os.path.join(BASE,'data','interactions.csv')); services=pd.read_csv(os.path.join(BASE,'data','services.csv')); model=HybridRecommender(); model.fit(customers, interactions, services); os.makedirs(os.path.join(BASE,'models'), exist_ok=True); model.save(os.path.join(BASE,'models','torco_recommender.pkl'))"

# Stage 2: Runtime image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY --from=builder /app /app

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir pandas numpy scikit-learn scipy joblib fastapi uvicorn pydantic httpx

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
