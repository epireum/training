#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file and add your GEMINI_API_KEY"
    exit 1
fi

# Build and start the multi-agent system
echo "Building and starting Google Agent system..."
docker-compose up --build
