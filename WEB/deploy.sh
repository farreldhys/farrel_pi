#!/bin/bash

# Configuration
IMAGE_NAME="pahlawan-app"
CONTAINER_NAME="pahlawan-app-container"
PORT=8053

echo "🚀 Starting deployment process..."

# Step 1: Build the Docker image
echo "📦 Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# Step 2: Stop and remove existing container if it exists
if [ $(docker ps -a -q -f name=$CONTAINER_NAME) ]; then
    echo "🛑 Stopping existing container..."
    docker stop $CONTAINER_NAME
    echo "🗑️ Removing existing container..."
    docker rm $CONTAINER_NAME
fi

# Step 3: Run the new container
echo "🏃‍♂️ Running new container on port $PORT..."
docker run -d -p $PORT:$PORT --name $CONTAINER_NAME $IMAGE_NAME

echo "✅ Deployment successful!"
echo "🌐 Application is running at port $PORT"
