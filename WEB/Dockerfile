# Use official Python 3.12.3 image as base
FROM python:3.12.3-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8053
EXPOSE 8053

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8053", "--server.address=0.0.0.0"]
