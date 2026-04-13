# Use official Python lightweight image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for compiling certain python packages like Chroma/hnswlib)
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the uploads directory and give permissions
RUN mkdir -p /app/uploads && chmod 777 /app/uploads
RUN mkdir -p /app/chroma_db && chmod 777 /app/chroma_db

# Copy the rest of the application code
COPY . .

# Expose port 7860 (Hugging Face Spaces default requirement)
EXPOSE 7860

# Run the FastAPI server via Uvicorn (Hugging Face routes to 7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
