# Use a slim Python base image
FROM python:3.10-slim

# 1. Install system dependencies
# We add libgomp1 for LightGBM/XGBoost/Catboost
# We keep chromium/chromium-driver for Selenium
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    chromium \
    chromium-driver \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set the working directory
WORKDIR /app

# 4. Copy requirements and install dependencies
# Using --system to install into the image's global python
COPY requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt

# 5. Copy EVERYTHING (scripts, json, and ensemble_model folder)
COPY . .

# 6. Set the entry point to your main script
CMD ["python", "daily_bbref_nb.py"]

