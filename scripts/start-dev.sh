#!/bin/bash

# Development startup script for Call Analysis System

set -e

echo "🚀 Starting Call Analysis System in Development Mode"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before continuing"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start development services
echo "🔨 Building development containers..."
docker-compose --profile dev build

echo "🗄️  Starting database and cache services..."
docker-compose up -d postgres redis

echo "⏳ Waiting for services to be healthy..."
docker-compose exec postgres pg_isready -U postgres
docker-compose exec redis redis-cli ping

echo "🗃️  Running database migrations..."
docker-compose --profile migration run --rm migration

echo "🌐 Starting API server in development mode..."
docker-compose --profile dev up api-dev

echo "✅ Development environment is ready!"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"