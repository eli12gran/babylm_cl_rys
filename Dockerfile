FROM pytorch/pytorch:2.0.1-cuda11.8-runtime-ubuntu22.04
 
WORKDIR /workspace
 
# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy all training code
COPY train.py .
COPY tokenizer_utilities.py .
COPY handler.py .
 
# Create data directory
RUN mkdir -p /workspace/data
 
# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH
 
# For local development: run train.py directly
# For RunPod serverless: the endpoint will call handler.py
CMD ["python", "train.py"]