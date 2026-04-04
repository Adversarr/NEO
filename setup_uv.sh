#!/bin/bash

# G2PT UV Environment Setup Script
# This script sets up the environment using UV package manager

set -euo pipefail

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if UV is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        log_error "UV is not installed or not in PATH"
        log_info "Please install UV first: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
}

# Main setup function
main() {
    log_info "Setting up G2PT environment with UV..."
    
    # Check UV installation
    check_uv
    
    # Sync dependencies including PyG group
    log_info "Syncing dependencies with PyG group..."
    if uv sync --group=pg; then
        log_success "Dependencies synced successfully"
    else
        log_error "Failed to sync dependencies"
        exit 1
    fi
    
    # Install package in editable mode
    log_info "Installing G2PT package in editable mode..."
    if uv pip install -e .; then
        log_success "G2PT package installed successfully"
    else
        log_error "Failed to install G2PT package"
        exit 1
    fi
    
    log_success "Environment setup completed!"
    log_info "You can now activate the environment with: source .venv/bin/activate"
}

# Run main function
main "$@"