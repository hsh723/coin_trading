FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create logs directory
RUN mkdir -p /app/logs

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_ENV=production \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    LOG_DIR=/app/logs \
    PROMETHEUS_PORT=9090 \
    GRAFANA_PORT=3000

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "src/main.py"]