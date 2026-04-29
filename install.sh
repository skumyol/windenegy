#!/bin/bash
# Windenegy Installation Script
# Downloads data, trains models, and prepares for Docker deployment

set -e

echo "=================================="
echo "Windenegy Installation"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_warn "Docker not found. You'll need Docker to run the application"
    fi
    
    # Check for Alpine Linux and install build tools
    if [ -f "/etc/alpine-release" ]; then
        log_info "Alpine Linux detected, installing build dependencies..."
        if ! command -v gcc &> /dev/null; then
            apk add --no-cache gcc g++ musl-dev linux-headers cmake make libgomp
        fi
    fi
    
    if ! command -v uv &> /dev/null; then
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    log_info "Prerequisites OK"
}

# Download SCADA data
download_data() {
    log_info "Checking for SCADA data..."
    
    if [ -f "data/raw/T1.csv" ]; then
        log_warn "Data already exists at data/raw/T1.csv"
        read -p "Re-download? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Using existing data"
            return
        fi
    fi
    
    mkdir -p data/raw
    
    # Auto-download using the script
    log_info "Attempting to download dataset..."
    if command -v kaggle &> /dev/null; then
        uv run python scripts/download_scada.py
    else
        log_warn "Kaggle CLI not found. Installing..."
        uv pip install kaggle
        uv run python scripts/download_scada.py
    fi
    
    # Check if download succeeded
    if [ ! -f "data/raw/T1.csv" ]; then
        log_error "Auto-download failed. Please download manually:"
        log_info "https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset"
        log_info "Place T1.csv in data/raw/"
        exit 1
    fi
    
    log_info "Data ready at data/raw/T1.csv"
}

# Setup Python environment
setup_environment() {
    log_info "Setting up Python environment..."
    
    if [ ! -d ".venv" ]; then
        uv venv
    fi
    
    source .venv/bin/activate
    uv pip install -e "."
    
    log_info "Environment ready"
}

# Train all models
train_models() {
    log_info "Training models..."
    log_info "This may take 5-10 minutes..."
    
    mkdir -p artifacts/models artifacts/metrics
    
    # Train gradient boosting models
    log_info "Training Gradient Boosting models..."
    uv run python scripts/train_all_models.py
    
    # Train PatchTST models
    log_info "Training PatchTST models..."
    uv run python scripts/train_patchtst.py --horizon 1
    uv run python scripts/train_patchtst.py --horizon 6
    uv run python scripts/train_patchtst.py --horizon 24
    
    log_info "All models trained successfully"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    model_count=$(find artifacts/models -name "metadata.json" | wc -l)
    
    if [ "$model_count" -lt 6 ]; then
        log_error "Expected 6 models, found $model_count"
        exit 1
    fi
    
    log_info "Found $model_count trained models"
    log_info "Models:"
    find artifacts/models -name "metadata.json" -exec dirname {} \; | xargs -n1 basename | sed 's/^/  - /'
}

# Build Docker image
build_docker() {
    log_info "Building Docker image..."
    
    if ! command -v docker &> /dev/null; then
        log_warn "Docker not available, skipping image build"
        return
    fi
    
    docker build -t windenegy:latest .
    
    log_info "Docker image built: windenegy:latest"
}

# Main installation flow
main() {
    check_prerequisites
    download_data
    setup_environment
    train_models
    verify_installation
    build_docker
    
    echo ""
    echo "=================================="
    log_info "Installation complete!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run locally:    ./run_dev.sh"
    echo "  2. Run with Docker: ./run_prod.sh start"
    echo "  3. Deploy to VPS:   git push origin main"
    echo ""
}

main "$@"
