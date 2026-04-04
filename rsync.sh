#!/bin/bash

set -euo pipefail

# Default configuration
RSYNC_OPTIONS="-avz --delete"
DRY_RUN=false
VERBOSE=true
BACKUP=false
SSH_PORT=22
SSH_KEY=""
EXCLUDE_PATTERNS=(
  "*.log"
  "*.tmp"
  ".git"
  "__pycache__"
  "*.pyc"
  ".DS_Store"
  "node_modules/"
  "venv/"
  "env/"
)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help information
usage() {
  cat << EOF
Usage: $0 [OPTIONS] <target_server> [target_path]

Sync current directory to remote server using rsync.

Arguments:
  target_server    Remote server (user@host)
  target_path      Remote path (default: ~/$(basename $(pwd)))

Options:
  -h, --help          Show this help message
  -n, --dry-run       Perform a trial run with no changes
  -q, --quiet         Suppress non-error output
  -b, --backup        Create backups of existing files on target
  -p, --port PORT     SSH port (default: 22)
  -i, --identity KEY  SSH private key file
  -e, --exclude PATTERN Additional exclude pattern (can be used multiple times)
  --progress          Show progress during transfer
  --no-compress       Disable compression
  --no-delete         Don't delete files on target that don't exist locally

Examples:
  $0 user@server
  $0 user@server /opt/project
  $0 -n -p 2222 -i ~/.ssh/id_rsa user@server:/var/www/html
  $0 -e "*.cache" -e "temp/" user@server
  $0 --progress large_directory user@server

EOF
  exit 0
}

log() {
  if [[ "$VERBOSE" == true ]]; then
    echo -e "${BLUE}[INFO]${NC} $1"
  fi
}

warn() {
  echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

error() {
  echo -e "${RED}[ERROR]${NC} $1" >&2
  exit 1
}

success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      ;;
    -n|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -q|--quiet)
      VERBOSE=false
      shift
      ;;
    -b|--backup)
      BACKUP=true
      shift
      ;;
    -p|--port)
      SSH_PORT="$2"
      shift 2
      ;;
    -i|--identity)
      SSH_KEY="$2"
      shift 2
      ;;
    -e|--exclude)
      EXCLUDE_PATTERNS+=("$2")
      shift 2
      ;;
    --progress)
      RSYNC_OPTIONS+=" --progress"
      shift
      ;;
    --no-compress)
      RSYNC_OPTIONS="${RSYNC_OPTIONS//-z/}"
      shift
      ;;
    --no-delete)
      RSYNC_OPTIONS="${RSYNC_OPTIONS//--delete/}"
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Get positional arguments
TARGET_SERVER="${1:-}"
TARGET_PATH="${2:-}"

# Verify arguments
if [[ -z "$TARGET_SERVER" ]]; then
  error "Target server is required. Use -h for help."
fi

# Set default target path
if [[ -z "$TARGET_PATH" ]]; then
  TARGET_PATH="~/$(basename "$(pwd)")"
fi

# Build complete remote path
if [[ "$TARGET_PATH" =~ ^/ ]]; then
  REMOTE_PATH="$TARGET_SERVER:$TARGET_PATH"
else
  REMOTE_PATH="$TARGET_SERVER:$TARGET_PATH"
fi

# Build SSH options
SSH_CMD="ssh"
if [[ -n "$SSH_PORT" && "$SSH_PORT" != "22" ]]; then
  SSH_CMD+=" -p $SSH_PORT"
fi
if [[ -n "$SSH_KEY" ]]; then
  if [[ ! -f "$SSH_KEY" ]]; then
    error "SSH key file not found: $SSH_KEY"
  fi
  SSH_CMD+=" -i $SSH_KEY"
fi

# Build exclude options
EXCLUDE_OPTS=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
  EXCLUDE_OPTS+=(--exclude="$pattern")
done

# Check required files
if [[ ! -f ".gitignore" ]]; then
  warn ".gitignore file not found, skipping --exclude-from"
else
  EXCLUDE_OPTS+=(--exclude-from='.gitignore')
fi

# Add backup options
if [[ "$BACKUP" == true ]]; then
  RSYNC_OPTIONS+=" --backup --backup-dir=.rsync_backup_$(date +%Y%m%d_%H%M%S)"
fi

# Add dry-run options
if [[ "$DRY_RUN" == true ]]; then
  RSYNC_OPTIONS+=" --dry-run"
  log "Performing dry run (no changes will be made)"
fi

# Add SSH options
RSYNC_OPTIONS+=" -e '$SSH_CMD'"

# Display configuration information
if [[ "$VERBOSE" == true ]]; then
  echo "========================================"
  log "Starting rsync operation"
  log "Remote server: $TARGET_SERVER"
  log "Remote path: $TARGET_PATH"
  log "SSH port: $SSH_PORT"
  [[ -n "$SSH_KEY" ]] && log "SSH key: $SSH_KEY"
  log "Rsync options: $RSYNC_OPTIONS"
  log "Exclude patterns: ${EXCLUDE_PATTERNS[*]}"
  [[ "$DRY_RUN" == true ]] && log "DRY RUN MODE ENABLED"
  [[ "$RSYNC_OPTIONS" =~ --progress ]] && log "Progress display: ENABLED"
  echo "========================================"
fi

# Execute rsync
log "Syncing $(pwd) to $REMOTE_PATH"
if eval rsync $RSYNC_OPTIONS "${EXCLUDE_OPTS[@]}" . "$REMOTE_PATH"; then
  success "Sync completed successfully"
  if [[ "$DRY_RUN" == true ]]; then
    warn "This was a dry run. No actual changes were made."
  fi
else
  error "Sync failed"
fi