# Use lightweight base image
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code and model
COPY ./app /app
COPY ./model /app/model
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Entry point
CMD ["python", "extract_outline.py"]
