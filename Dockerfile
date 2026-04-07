FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment variables (overridden at runtime)
ENV API_BASE_URL=https://api.groq.com/openai/v1
ENV MODEL_NAME=llama-3.3-70b-versatile
ENV HF_TOKEN=""
ENV PORT=7860

# Expose port for FastAPI
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default: run the FastAPI server
# Override with: docker run ... python inference.py
CMD ["python", "app.py"]
