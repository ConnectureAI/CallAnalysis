#!/bin/bash

# Production deployment script for Call Analysis System

set -e

echo "🚀 Deploying Call Analysis System to Production"

# Validate required environment variables
required_vars=("OPENAI_API_KEY" "SECURITY_SECRET_KEY" "DB_PASSWORD")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set"
        exit 1
    fi
done

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Pull latest images and build
echo "📦 Building production containers..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile prod build --no-cache

# Start infrastructure services first
echo "🗄️  Starting infrastructure services..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d postgres redis

# Wait for services to be ready
echo "⏳ Waiting for infrastructure services..."
timeout 60 bash -c 'until docker-compose exec postgres pg_isready -U postgres; do sleep 2; done'
timeout 30 bash -c 'until docker-compose exec redis redis-cli ping; do sleep 2; done'

# Run database migrations
echo "🗃️  Running database migrations..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile migration run --rm migration

# Start application services
echo "🌐 Starting application services..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile prod up -d api worker

# Start reverse proxy
echo "🔒 Starting reverse proxy..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile prod up -d nginx

# Health check
echo "🔍 Performing health checks..."
sleep 10
timeout 30 bash -c 'until curl -f http://localhost:8000/health; do sleep 2; done'

echo "✅ Production deployment completed successfully!"
echo "🌐 API is available at: http://localhost"
echo "📊 Health endpoint: http://localhost/health"

# Display service status
echo "📋 Service Status:"
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile prod ps