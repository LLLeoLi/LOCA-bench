#!/bin/bash
set -euo pipefail

# ====== Config ======
# Determine paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
PROJECT_ROOT="${PROJECT_DIR}"

# Node config
NVM_VERSION="${NVM_VERSION:-v0.40.3}"
NODE_MAJOR="${NODE_MAJOR:-24}"

export DEBIAN_FRONTEND=noninteractive

# ====== Helpers ======
log() { echo -e "\n[install] $*\n"; }

# ====== 0) System dependencies ======
# Install python3-dev for building C extensions (e.g., pycosat)
log "Checking system dependencies..."

# Detect Python version for the correct -dev package
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

# The dependency pins in pyproject.toml are resolved for Python 3.12 (numpy and
# pandas both require >=3.12, and mcp_convert hard-requires >=3.12). Bail out now
# rather than let pip resolve a silently different set on the wrong interpreter.
if [[ "$PYTHON_VERSION" != "3.12" ]]; then
  log "ERROR: LOCA-bench requires Python 3.12, but python3 is ${PYTHON_VERSION}."
  log "Create and activate a 3.12 environment first, e.g.:"
  log "  conda create -n loca python=3.12 -y && conda activate loca"
  exit 1
fi

install_python_dev() {
  if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    log "Installing python${PYTHON_VERSION}-dev (required for building C extensions)..."
    sudo apt-get update -qq
    sudo apt-get install -y python${PYTHON_VERSION}-dev build-essential
  elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    log "Installing python3-devel (required for building C extensions)..."
    sudo yum install -y python3-devel gcc
  elif command -v dnf &> /dev/null; then
    # Fedora
    log "Installing python3-devel (required for building C extensions)..."
    sudo dnf install -y python3-devel gcc
  else
    log "WARNING: Could not detect package manager. Please install python3-dev manually."
  fi
}

# Check if Python.h exists
PYTHON_INCLUDE=$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
if [[ ! -f "${PYTHON_INCLUDE}/Python.h" ]]; then
  log "Python development headers not found at ${PYTHON_INCLUDE}"
  if command -v sudo &> /dev/null; then
    install_python_dev
  else
    log "ERROR: Python development headers missing and no sudo access."
    log "Please ask your administrator to install: python${PYTHON_VERSION}-dev"
    exit 1
  fi
else
  log "Python development headers found at ${PYTHON_INCLUDE}"
fi

# ====== 1) Python deps ======
# Use uv if available (faster), otherwise fall back to pip
if command -v uv &> /dev/null; then
  PIP_CMD="uv pip"
  log "Installing Python dependencies (uv)..."
else
  PIP_CMD="python -m pip"
  log "Installing Python dependencies (pip)..."
  python -m pip install --upgrade pip
fi

# Pre-install common deps MCP servers need.
# Every version here is pinned and must stay in sync with the pins in
# pyproject.toml -- floating specs pull breaking releases on fresh nodes.
$PIP_CMD install --no-cache-dir \
  "fire==0.7.1" \
  "python-dotenv==1.2.2" \
  "tiktoken==0.13.0" \
  "uv==0.12.0" \
  "reportlab==5.0.0" \
  "cryptography==49.0.0" \
  "ruff==0.16.0" \
  "black==26.5.1" \
  "pandas==3.0.5" \
  "numpy==2.5.1" \
  "pydantic-core==2.46.4" \
  "openpyxl==3.1.5" \
  "pillow==12.3.0"

# Install the local project in editable mode (includes fastmcp, excel-mcp-server, etc.)
if [[ -d "$PROJECT_DIR" ]]; then
  log "Installing local project editable: $PROJECT_DIR"
  (cd "$PROJECT_DIR" && $PIP_CMD install -e .)
else
  log "WARNING: Project directory not found: $PROJECT_DIR"
  exit 1
fi

# ====== 1b) Pre-build mcp_convert uv environment ======
# uv-based MCP servers (google_cloud, calendar, google_sheet, snowflake,
# woocommerce) are launched at eval time via `uv --directory mcp_convert run`.
# Building the env now (managed Python >=3.12 download + deps from uv.lock)
# keeps first-task server startup well under the 120s MCP discovery timeout.
log "Pre-building mcp_convert uv environment..."
# --frozen matches how config_loader.py launches these servers (`uv run --frozen`),
# so the env built here is byte-for-byte the one used at eval time.
uv --directory "$PROJECT_DIR/mcp_convert" sync --frozen

# ====== 2) nvm + Node ======
log "Installing nvm ($NVM_VERSION) and Node.js ($NODE_MAJOR)..."
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"

# Install nvm if not already installed
if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
  log "Installing nvm..."
  curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" | bash

  if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
    log "ERROR: nvm installation failed"
    exit 1
  fi
fi

# Load nvm
# shellcheck source=/dev/null
. "$NVM_DIR/nvm.sh"

# Install and configure Node.js
log "Installing Node.js v$NODE_MAJOR..."
nvm install "$NODE_MAJOR"
nvm use "$NODE_MAJOR"
nvm alias default "$NODE_MAJOR"

# Verify installation
if ! command -v node &> /dev/null; then
  log "ERROR: Node.js installation failed"
  exit 1
fi

# ====== 3) Create symlinks for easier access ======
log "Creating symlinks for node/npm/npx in ~/.local/bin..."
mkdir -p "$HOME/.local/bin"

# Dynamically get the actual installed node path
NODE_PATH="$(nvm which node)"
if [[ -z "$NODE_PATH" ]]; then
  log "ERROR: Could not determine node path"
  exit 1
fi

NODE_DIR="$(dirname "$NODE_PATH")"
log "Node.js installed at: $NODE_DIR"

# Create symlinks
ln -sf "$NODE_DIR/node" "$HOME/.local/bin/node"
ln -sf "$NODE_DIR/npm"  "$HOME/.local/bin/npm"
ln -sf "$NODE_DIR/npx"  "$HOME/.local/bin/npx"

# Update PATH for this session
export PATH="$HOME/.local/bin:$PATH"

# ====== 4) npm global packages ======
log "Installing npm global packages..."
# Pinned: these must match the versions in config/filesystem.yaml and
# config/memory.yaml so npx serves from cache instead of re-resolving.
npm install -g \
  @modelcontextprotocol/server-filesystem@2026.1.14 \
  @modelcontextprotocol/server-memory@2026.7.4

# ====== 5) Pre-cache uvx tools ======
# Warm the exact ephemeral environments the eval-time uvx invocations resolve
# (matching the `with_requirements` pins in gem/tools/mcp_server/config/*.yaml).
# </dev/null makes the stdio servers exit immediately after the env is built.
log "Pre-caching uvx tools (ignore failures)..."
uvx --help || true
ALLOWED_DIR=/tmp uvx --with "mcp==1.29.0" "cli-mcp-server==0.2.5" </dev/null || true
uvx --with "fastmcp==2.14.7" --with "mcp==1.29.0" "pdf-tools-mcp==0.1.4" </dev/null || true

log "Done ✅"
