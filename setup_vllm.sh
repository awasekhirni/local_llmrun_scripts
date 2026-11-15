#!/bin/bash
set -euo pipefail

# ==============================================================================
# vLLM Setup & Execution Script for Agentic AI Workflows
# Author: Awase Khirni Syed
# Description: Sets up vLLM, enforces 50% disk capacity allocation, and maps 
#              Ollama model names to Hugging Face IDs (or loads local GGUF).
# setup instructions 
#    chmod +x setup_vllm.sh
#    ./setup_vllm.sh -m qwen2.5
#Run with a local Ollama GGUF blob:
# If you already downloaded a model via Ollama and want to use the raw file:
#    ./setup_vllm.sh -m ~/.ollama/models/blobs/sha256-<your_hash>
# ==============================================================================

# --- Configuration ---
MODEL_DIR="${VLLM_MODEL_DIR:-$HOME/vllm_models}"
VENV_DIR="${VENV_DIR:-$HOME/vllm_venv}"
TARGET_PARTITION="/" # Change this if you want to enforce the 50% rule on a specific mount point

# --- Color Output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --- Usage ---
usage() {
    cat << EOF
Usage: $0 -m <model_name_or_path> [options]

Options:
  -m, --model       Ollama model name (e.g., 'llama3', 'qwen2.5') OR path to a local .gguf file.
  -p, --port        Port to run the vLLM server on (default: 8000).
  -c, --ctx-len     Maximum context length (default: 4096).
  -h, --help        Show this help message.

Examples:
  $0 -m llama3
  $0 -m qwen2.5 -p 8080
  $0 -m /path/to/local/ollama_model.gguf
EOF
    exit 0
}

# --- Argument Parsing ---
MODEL_INPUT=""
PORT=8000
CTX_LEN=4096

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL_INPUT="$2"; shift ;;
        -p|--port) PORT="$2"; shift ;;
        -c|--ctx-len) CTX_LEN="$2"; shift ;;
        -h|--help) usage ;;
        *) log_error "Unknown parameter passed: $1" ;;
    esac
    shift
done

if [[ -z "$MODEL_INPUT" ]]; then
    log_error "Model name or path is required. Use -m <model>."
fi

# ==============================================================================
# 1. Enforce 50% Disk Allocation Rule
# ==============================================================================
log_info "Verifying disk space allocation (Target: 50% of total capacity)..."

mkdir -p "$MODEL_DIR"

# Get total and available space in bytes
TOTAL_BYTES=$(df -B1 "$TARGET_PARTITION" | awk 'NR==2 {print $2}')
AVAIL_BYTES=$(df -B1 "$TARGET_PARTITION" | awk 'NR==2 {print $4}')
HALF_TOTAL=$((TOTAL_BYTES / 2))

if [[ "$AVAIL_BYTES" -lt "$HALF_TOTAL" ]]; then
    log_error "Available disk space is less than 50% of total capacity. 
    Total: $(numfmt --to=iec "$TOTAL_BYTES") | Available: $(numfmt --to=iec "$AVAIL_BYTES") | Required 50%: $(numfmt --to=iec "$HALF_TOTAL")"
fi

log_info "Disk allocation verified. $(numfmt --to=iec "$HALF_TOTAL") (50%) is available for vLLM."

# ==============================================================================
# 2. Map Ollama Model to Hugging Face (or detect local GGUF)
# ==============================================================================
VLLM_MODEL_ID=""
LOAD_FORMAT=""

if [[ "$MODEL_INPUT" == *.gguf ]]; then
    log_info "Detected local GGUF file. Configuring vLLM for GGUF load format."
    VLLM_MODEL_ID="$MODEL_INPUT"
    LOAD_FORMAT="--load-format gguf"
else
    log_info "Mapping Ollama model name '$MODEL_INPUT' to Hugging Face repository..."
    case "$MODEL_INPUT" in
        llama3)       VLLM_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct" ;;
        llama3.1)     VLLM_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
        llama3.2)     VLLM_MODEL_ID="meta-llama/Llama-3.2-3B-Instruct" ;;
        qwen2.5)      VLLM_MODEL_ID="Qwen/Qwen2.5-7B-Instruct" ;;
        qwen2.5:14b)  VLLM_MODEL_ID="Qwen/Qwen2.5-14B-Instruct" ;;
        mistral)      VLLM_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3" ;;
        phi3)         VLLM_MODEL_ID="microsoft/Phi-3-mini-4k-instruct" ;;
        *)            
            # If it's not a mapped Ollama name, assume it's already a valid HF ID
            VLLM_MODEL_ID="$MODEL_INPUT" 
            ;;
    esac
    log_info "Resolved model: $VLLM_MODEL_ID"
fi

# ==============================================================================
# 3. Environment & vLLM Setup
# ==============================================================================
export HF_HOME="$MODEL_DIR/huggingface"
export VLLM_USAGE_SOURCE="agentic-workflow"

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. vLLM requires NVIDIA GPUs and CUDA drivers."
fi

# Setup Python Virtual Environment
if [[ ! -d "$VENV_DIR" ]]; then
    log_info "Creating Python virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install/Upgrade vLLM and HuggingFace CLI
log_info "Ensuring vLLM and dependencies are installed..."
pip install --quiet --upgrade pip
pip install --quiet vllm huggingface_hub

# ==============================================================================
# 4. Launch vLLM Server
# ==============================================================================
log_info "Starting vLLM OpenAI-compatible API server on port $PORT..."
log_info "Model: $VLLM_MODEL_ID | Context Length: $CTX_LEN"

# Construct the command
CMD="vllm serve $VLLM_MODEL_ID \
    --host 0.0.0.0 \
    --port $PORT \
    --max-model-len $CTX_LEN \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    $LOAD_FORMAT"

# Execute
eval $CMD
