#!/bin/bash
set -e

# Windenegy Production Runner
# Usage: ./run_prod.sh [command] [--check]
# Commands:
#   up          - Start all services with Docker Compose
#   down        - Stop all services
#   build       - Build Docker images
#   logs        - View service logs
#   --check     - Check if services are running

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${WINDENEGY_API_PORT:-8765}"
DASHBOARD_PORT="${WINDENEGY_DASHBOARD_PORT:-8766}"

cd "$PROJECT_ROOT"

# Check if Docker and Docker Compose are available
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo "Error: Docker Compose is not installed"
        exit 1
    fi
}

# Get the correct docker compose command
get_compose_cmd() {
    if docker compose version &> /dev/null; then
        echo "docker compose"
    else
        echo "docker-compose"
    fi
}

check_servers() {
    echo "Checking service status..."
    
    local healthy=true
    
    # Check API
    if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        echo "✓ API healthy at http://localhost:$API_PORT"
    else
        echo "✗ API not responding at http://localhost:$API_PORT"
        healthy=false
    fi
    
    # Check Dashboard
    if curl -s "http://localhost:$DASHBOARD_PORT" > /dev/null 2>&1; then
        echo "✓ Dashboard healthy at http://localhost:$DASHBOARD_PORT"
    else
        echo "✗ Dashboard not responding at http://localhost:$DASHBOARD_PORT"
        healthy=false
    fi
    
    if [ "$healthy" = true ]; then
        echo ""
        echo "All services are running!"
        exit 0
    else
        exit 1
    fi
}

start_services() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    echo "Starting Windenegy services..."
    $compose_cmd up -d
    
    echo ""
    echo "Waiting for services to be ready..."
    sleep 5
    
    check_servers
}

stop_services() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    echo "Stopping Windenegy services..."
    $compose_cmd down
    echo "Services stopped."
}

build_images() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    echo "Building Docker images..."
    $compose_cmd build --no-cache
    echo "Build complete."
}

view_logs() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    echo "Viewing logs (Ctrl+C to exit)..."
    $compose_cmd logs -f
}

# Main command dispatcher
case "${1:-}" in
    up)
        start_services
        ;;
    down)
        stop_services
        ;;
    build)
        build_images
        ;;
    logs)
        view_logs
        ;;
    --check)
        check_servers
        ;;
    *)
        echo "Windenegy Production Runner"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  up        Start all services with Docker Compose"
        echo "  down      Stop all services"
        echo "  build     Build Docker images"
        echo "  logs      View service logs"
        echo "  --check   Check if services are running"
        echo ""
        echo "Environment variables:"
        echo "  WINDENEGY_API_PORT        API server port (default: 8000)"
        echo "  WINDENEGY_DASHBOARD_PORT  Dashboard port (default: 8501)"
        echo ""
        echo "Examples:"
        echo "  $0 up                    # Start all services"
        echo "  $0 --check               # Check service health"
        echo "  $0 down                  # Stop all services"
        exit 1
        ;;
esac
