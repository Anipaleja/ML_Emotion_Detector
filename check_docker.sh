#!/bin/bash

# check_docker.sh - Script to check Docker status and wait for it to be ready

# Make the script exit on error
set -e

echo "Checking Docker status..."

# Function to check if Docker daemon is running
check_docker_daemon() {
  echo "Checking if Docker daemon is running..."
  
  # Try to run a simple docker command
  if docker info &> /dev/null; then
    echo "✅ Docker daemon is running."
    return 0
  else
    echo "❌ Docker daemon is not running."
    echo "Please start Docker Desktop and try again."
    return 1
  fi
}

# Function to check if Docker Compose is available
check_docker_compose() {
  echo "Checking if Docker Compose is available..."
  
  # First try with docker compose (V2)
  if docker compose version &> /dev/null; then
    echo "✅ Docker Compose V2 is available."
    return 0
  # Then try with docker-compose (V1)
  elif docker-compose --version &> /dev/null; then
    echo "✅ Docker Compose V1 is available."
    return 0
  else
    echo "❌ Docker Compose is not available."
    echo "Please make sure Docker Compose is installed correctly."
    return 1
  fi
}

# Function to show Docker version information
show_docker_info() {
  echo "Docker version information:"
  echo "-------------------------"
  docker --version
  
  # Check which version of Docker Compose is available
  if docker compose version &> /dev/null; then
    docker compose version
  elif docker-compose --version &> /dev/null; then
    docker-compose --version
  fi
  
  echo "-------------------------"
  echo "Docker system info:"
  docker info | grep -E 'Server Version|Docker Root Dir|Debug Mode|Containers|Images'
}

# Main function
main() {
  # Wait for Docker daemon to be available
  max_retries=30
  retry_count=0
  
  while ! check_docker_daemon && [ $retry_count -lt $max_retries ]; do
    echo "Waiting for Docker daemon to start..."
    sleep 2
    retry_count=$((retry_count+1))
  done
  
  if [ $retry_count -eq $max_retries ]; then
    echo "Timed out waiting for Docker daemon to start."
    echo "Please make sure Docker Desktop is running."
    exit 1
  fi
  
  # Check Docker Compose
  if ! check_docker_compose; then
    exit 1
  fi
  
  # Show Docker information
  show_docker_info
  
  echo ""
  echo "Docker is ready! ✅"
  echo "You can now run: docker-compose -f docker/docker-compose.yml build"
}

# Run the main function
main

