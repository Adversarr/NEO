#!/bin/bash

# ==============================================================================
# G2PT Environment Setup Script
# ==============================================================================
# This script sets up the conda environment for the G2PT project with proper
# error handling, logging, and best practices.
#
# Usage: ./setup_conda.sh [options]
# Options:
#   -n, --name NAME      Environment name (default: g2pt)
#   -p, --python VERSION Python version (default: 3.12)
#   -f, --force          Recreate environment if it exists
#   -h, --help           Show this help message
#
# Examples:
#   ./setup_conda.sh                    # Create default 'g2pt' environment
#   ./setup_conda.sh -n my_env -f       # Recreate 'my_env' environment
# ==============================================================================

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default configuration values
DEFAULT_ENV_NAME="g2pt"
DEFAULT_PYTHON_VERSION="3.12"
REQUIREMENTS_FILES=("requirements.txt" "requirements-pyg.txt")

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

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

# Cleanup function to handle script termination
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Script failed with exit code $exit_code"
        log_info "Cleaning up temporary files..."
        # Add any cleanup operations here
    fi
}
trap cleanup EXIT

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if conda is available
check_conda() {
    if ! command_exists conda; then
        log_error "conda is not installed or not in PATH"
        log_info "Please install Anaconda or Miniconda first"
        exit 1
    fi
}

# Check if files exist
check_files() {
    local missing_files=()
    for file in "$@"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        log_error "The following required files are missing:"
        printf '%s\n' "${missing_files[@]}"
        exit 1
    fi
}

# Check if conda environment exists
env_exists() {
    local env_name=$1
    conda env list | grep -q "^$env_name\s"
}

# Parse command line arguments
parse_args() {
    ENV_NAME="$DEFAULT_ENV_NAME"
    PYTHON_VERSION="$DEFAULT_PYTHON_VERSION"
    FORCE_RECREATE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--name)
                ENV_NAME="$2"
                shift 2
                ;;
            -p|--python)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_RECREATE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Show help message
show_help() {
    cat << EOF
$(basename "$0") - G2PT Environment Setup Script

This script creates and configures a conda environment for the G2PT project.

USAGE:
    $(basename "$0") [OPTIONS]

OPTIONS:
    -n, --name NAME       Environment name (default: $DEFAULT_ENV_NAME)
    -p, --python VERSION  Python version (default: $DEFAULT_PYTHON_VERSION)
    -f, --force           Recreate environment if it already exists
    -h, --help           Show this help message

EXAMPLES:
    $(basename "$0")                    # Create default environment
    $(basename "$0") -n my_env          # Create 'my_env' environment
    $(basename "$0") -f                 # Recreate default environment
    $(basename "$0") -n my_env -f       # Recreate 'my_env' environment

EOF
}

# Show brief usage
show_usage() {
    echo "Usage: $(basename "$0") [-n NAME] [-p VERSION] [-f] [-h]"
    echo "Try '$(basename "$0") --help' for more information."
}

# ==============================================================================
# MAIN SETUP FUNCTIONS
# ==============================================================================

# Check if running in non-interactive mode
is_non_interactive() {
    [[ ! -t 0  ]]
}

# Create or recreate conda environment
setup_conda_env() {
    local env_name=$1
    local python_version=$2
    local force=$3
    
    if env_exists "$env_name"; then
        if [ "$force" = true ]; then
            log_warning "Environment '$env_name' already exists. Removing..."
            conda env remove -n "$env_name" -y
            log_success "Environment '$env_name' removed"
        else
            log_warning "Environment '$env_name' already exists."
            
            if is_non_interactive; then
                log_info "Running in non-interactive mode. Use existing environment."
                log_info "To force recreate, run with -f flag: $0 -f"
            else
                echo
                echo "Options:"
                echo "  [1] Force recreate environment (delete and recreate)"
                echo "  [2] Use existing environment (continue with setup)"
                echo "  [q] Quit"
                echo
                
                while true; do
                    read -p "Please select an option (1/2/q): " choice
                    case "$choice" in
                        1)
                            log_warning "Removing existing environment '$env_name'..."
                            conda env remove -n "$env_name" -y
                            log_success "Environment '$env_name' removed"
                            break
                            ;;
                        2)
                            log_info "Using existing environment '$env_name'"
                            log_info "Skipping environment creation, continuing with package installation..."
                            return 0
                            ;;
                        q|Q)
                            log_info "User cancelled the operation"
                            exit 0
                            ;;
                        *)
                            log_error "Invalid option: '$choice'"
                            ;;
                    esac
                done
            fi
        fi
    fi
    
    log_info "Creating conda environment '$env_name' with Python $python_version..."
    conda create -n "$env_name" python="$python_version" -y
    
    log_success "Conda environment '$env_name' created successfully"
    
    # Verify actual Python version installed
    local actual_python_version
    actual_python_version=$(conda run -n "$env_name" python --version 2>&1 | cut -d' ' -f2)
    log_info "Actual Python version installed: $actual_python_version"
    
    # Check if it matches the requested version
    if [[ "$actual_python_version" == "$python_version"* ]]; then
        log_success "Python version matches requested version (prefix: $python_version)"
    else
        log_warning "Python version may not match requested version!"
        log_warning "Requested: $python_version.x, Got: $actual_python_version"
    fi
}

# Install PyTorch Geometric dependencies
install_pyg_dependencies() {
    local env_name=$1
    
    log_info "Installing PyTorch Geometric dependencies..."
    
    # Install PyG dependencies using conda run to activate environment
    conda run -n "$env_name" pip install --no-cache-dir -r requirements-pyg.txt
    
    log_success "PyTorch Geometric dependencies installed"
}

# Install main requirements
install_requirements() {
    local env_name=$1
    
    log_info "Installing main requirements..."
    
    # Install main requirements
    conda run -n "$env_name" pip install --no-cache-dir -r requirements.txt
    
    log_success "Main requirements installed"
}

# Install the package in development mode
install_package() {
    local env_name=$1
    
    log_info "Installing G2PT package in development mode..."
    
    # Install the package
    conda run -n "$env_name" pip install --no-cache-dir -e .
    
    log_success "G2PT package installed"
}

# Verify installation
verify_installation() {
    local env_name=$1
    
    log_info "Verifying installation..."
    
    # Test Python import
    if conda run -n "$env_name" python -c "import g2pt" 2>/dev/null; then
        log_success "G2PT package can be imported successfully"
    else
        log_error "G2PT package import failed"
        exit 1
    fi
    
    # Test key dependencies
    local key_deps=("torch" "torch_geometric" "lightning" "hydra")
    for dep in "${key_deps[@]}"; do
        if conda run -n "$env_name" python -c "import $dep" 2>/dev/null; then
            log_success "Dependency '$dep' is available"
        else
            log_warning "Dependency '$dep' is not available"
        fi
    done
}

# Show next steps
show_next_steps() {
    local env_name=$1
    
    echo
    log_success "Setup completed successfully!"
    echo
    log_info "Next steps:"
    echo "  1. Activate the environment:"
    echo "     conda activate $env_name"
    echo
    echo "  2. Verify the installation:"
    echo "     python -c 'import g2pt; print(\"G2PT installed successfully!\")'"
    echo
    echo "  3. Run tests (if available):"
    echo "     python -m pytest tests/ -v"
    echo
    echo "  4. Start using G2PT:"
    echo "     # Example training command"
    echo "     python exp/pretrain/train.py"
    echo
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

main() {
    # Parse command line arguments
    parse_args "$@"
    
    # Show configuration
    echo "====================================================================="
    log_info "G2PT Environment Setup"
    echo "====================================================================="
    log_info "Environment name: $ENV_NAME"
    log_info "Python version: $PYTHON_VERSION"
    log_info "Force recreate: $FORCE_RECREATE"
    echo "====================================================================="
    echo
    
    # Pre-flight checks
    check_conda
    check_files "${REQUIREMENTS_FILES[@]}" "pyproject.toml"
    
    # Main setup steps
    setup_conda_env "$ENV_NAME" "$PYTHON_VERSION" "$FORCE_RECREATE"
    install_requirements "$ENV_NAME"
    install_pyg_dependencies "$ENV_NAME"
    install_package "$ENV_NAME"
    verify_installation "$ENV_NAME"
    show_next_steps "$ENV_NAME"
}

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

# Run main function with all passed arguments
main "$@"