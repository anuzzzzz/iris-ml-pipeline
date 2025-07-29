FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for API
RUN pip install --no-cache-dir flask gunicorn

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY configs/ ./configs/
COPY dvc.yaml .
COPY dvc.lock .

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=src/api.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port (8081 to avoid conflict with Jupyter)
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Run the application with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8081", "--workers", "2", "--timeout", "60", "--keep-alive", "10", "src.api:app"]