#!/bin/bash
set -e

# Windenegy Development Runner
# Usage: ./run_dev.sh [command]
# Commands:
#   api       - Start FastAPI development server
#   dashboard - Start Streamlit dashboard
#   test      - Run test suite
#   lint      - Run linting and type checks
#   install   - Install dependencies
#   --check   - Check if servers are running

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${WINDENEGY_API_PORT:-8765}"
DASHBOARD_PORT="${WINDENEGY_DASHBOARD_PORT:-8766}"

cd "$PROJECT_ROOT"

# Check if uv is available, otherwise use pip
if command -v uv &> /dev/null; then
    PACKAGE_MANAGER="uv pip"
    RUN_PREFIX="uv run"
else
    PACKAGE_MANAGER="pip"
    RUN_PREFIX=""
fi

check_servers() {
    echo "Checking server status..."
    
    # Check API
    if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        echo "✓ API running on port $API_PORT"
    else
        echo "✗ API not running on port $API_PORT"
    fi
    
    # Check Dashboard
    if curl -s "http://localhost:$DASHBOARD_PORT" > /dev/null 2>&1; then
        echo "✓ Dashboard running on port $DASHBOARD_PORT"
    else
        echo "✗ Dashboard not running on port $DASHBOARD_PORT"
    fi
}

install_deps() {
    echo "Installing dependencies..."
    if command -v uv &> /dev/null; then
        uv pip install -e ".[dev,all]"
    else
        pip install -e ".[dev,all]"
    fi
    echo "Dependencies installed."
}

run_tests() {
    echo "Running tests..."
    if [ -f "pytest.ini" ] || [ -f "pyproject.toml" ]; then
        python -m pytest tests/ -v --tb=short
    else
        echo "No pytest configuration found."
        exit 1
    fi
}

run_lint() {
    echo "Running linting..."
    ruff check src tests
    ruff format --check src tests
    echo "Running type checks..."
    mypy src
    echo "All checks passed!"
}

start_api() {
    echo "Starting FastAPI development server..."
    export WINDENEGY_ENV=development
    export WINDENEGY_DEBUG=true
    export WINDENEGY_API_RELOAD=true
    
    if command -v uv &> /dev/null; then
        uv run uvicorn windenegy.interface.api:app --host 0.0.0.0 --port "$API_PORT" --reload
    else
        uvicorn windenegy.interface.api:app --host 0.0.0.0 --port "$API_PORT" --reload
    fi
}

start_dashboard() {
    echo "Starting Streamlit dashboard..."
    export WINDENEGY_ENV=development
    
    if command -v uv &> /dev/null; then
        uv run streamlit run src/windenegy/interface/dashboard.py --server.port "$DASHBOARD_PORT"
    else
        streamlit run src/windenegy/interface/dashboard.py --server.port "$DASHBOARD_PORT"
    fi
}

# Main command dispatcher
case "${1:-}" in
    api)
        start_api
        ;;
    dashboard)
        start_dashboard
        ;;
    test)
        run_tests
        ;;
    lint)
        run_lint
        ;;
    install)
        install_deps
        ;;
    --check)
        check_servers
        ;;
    *)
        echo "Windenegy Development Runner"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  api         Start FastAPI development server"
        echo "  dashboard   Start Streamlit dashboard"
        echo "  test        Run test suite"
        echo "  lint        Run linting and type checks"
        echo "  install     Install dependencies"
        echo "  --check     Check if servers are running"
        echo ""
        echo "Environment variables:"
        echo "  WINDENEGY_API_PORT        API server port (default: 8765)"
        echo "  WINDENEGY_DASHBOARD_PORT  Dashboard port (default: 8766)"
        exit 1
        ;;
esac
