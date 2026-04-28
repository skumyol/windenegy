#!/bin/bash
set -e

# Windenegy Development Runner
# Usage: ./run_dev.sh [command]
# Commands:
#   api       - Start FastAPI development server
#   dashboard - Start Streamlit dashboard
#   train     - Run unified training pipeline (all models, all horizons)
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

run_train() {
    echo "Running unified training pipeline..."
    if command -v uv &> /dev/null; then
        uv run python scripts/train_all.py "$@"
    else
        python scripts/train_all.py "$@"
    fi
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

free_port() {
    local port=$1
    local pids
    pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Port $port already in use by PID(s): $pids — killing..."
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
}

start_all() {
    echo "Starting API on port $API_PORT and dashboard on port $DASHBOARD_PORT..."
    echo "Press Ctrl+C to stop both."
    export WINDENEGY_ENV=development
    export WINDENEGY_DEBUG=true
    export WINDENEGY_API_RELOAD=true

    mkdir -p logs

    # Free ports if in use
    free_port "$API_PORT"
    free_port "$DASHBOARD_PORT"

    # Start API in background
    if command -v uv &> /dev/null; then
        uv run uvicorn windenegy.interface.api:app --host 0.0.0.0 --port "$API_PORT" --reload \
            > logs/api.log 2>&1 &
    else
        uvicorn windenegy.interface.api:app --host 0.0.0.0 --port "$API_PORT" --reload \
            > logs/api.log 2>&1 &
    fi
    API_PID=$!
    echo "  API     -> http://localhost:$API_PORT  (pid $API_PID, logs: logs/api.log)"

    # Start dashboard in background
    if command -v uv &> /dev/null; then
        uv run streamlit run src/windenegy/interface/dashboard.py \
            --server.port "$DASHBOARD_PORT" --server.headless true \
            > logs/dashboard.log 2>&1 &
    else
        streamlit run src/windenegy/interface/dashboard.py \
            --server.port "$DASHBOARD_PORT" --server.headless true \
            > logs/dashboard.log 2>&1 &
    fi
    DASH_PID=$!
    echo "  Dashboard -> http://localhost:$DASHBOARD_PORT  (pid $DASH_PID, logs: logs/dashboard.log)"

    # Cleanup on exit
    cleanup() {
        echo ""
        echo "Shutting down..."
        kill "$API_PID" "$DASH_PID" 2>/dev/null || true
        wait "$API_PID" "$DASH_PID" 2>/dev/null || true
        echo "Stopped."
    }
    trap cleanup INT TERM EXIT

    # Tail combined logs
    sleep 2
    echo ""
    echo "Streaming combined logs (Ctrl+C to stop both servers):"
    echo "-------------------------------------------------------"
    tail -F logs/api.log logs/dashboard.log
}

# Main command dispatcher
case "${1:-all}" in
    all)
        start_all
        ;;
    api)
        start_api
        ;;
    dashboard)
        start_dashboard
        ;;
    train)
        shift
        run_train "$@"
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
    -h|--help|help)
        echo "Windenegy Development Runner"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (none)|all  Start API + dashboard together (default)"
        echo "  api         Start FastAPI development server only"
        echo "  dashboard   Start Streamlit dashboard only"
        echo "  test        Run test suite"
        echo "  lint        Run linting and type checks"
        echo "  install     Install dependencies"
        echo "  --check     Check if servers are running"
        echo ""
        echo "Environment variables:"
        echo "  WINDENEGY_API_PORT        API server port (default: 8765)"
        echo "  WINDENEGY_DASHBOARD_PORT  Dashboard port (default: 8766)"
        exit 0
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 --help' for usage."
        exit 1
        ;;
esac
