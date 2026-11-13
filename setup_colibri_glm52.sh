#!/usr/bin/env bash
#
# run_glm52_colibri.sh
# ---------------------------------------------------------------------------
# End-to-end helper for running the GLM 5.2 model locally with the
# Colibri hierarchical inference engine (https://github.com/JustVugg/colibri).
#
# Colibri itself is a C engine that keeps hot MoE experts on the GPU,
# warm experts in RAM and cold experts streamed from NVMe. GLM 5.2 is a
# 744B-parameter MoE model, so this script also makes the disk / RAM
# requirements explicit before doing anything destructive.
#
# Quick start
#   ./run_glm52_colibri.sh check      # verify deps + hardware
#   ./run_glm52_colibri.sh install   # clone colibri + download glm-5.2
#   ./run_glm52_colibri.sh serve     # start OpenAI-compatible API on :8000
#   ./run_glm52_colibri.sh chat      # interactive chat
#   ./run_glm52_colibri.sh test      # curl the API with a GLM 5.2 prompt
#   ./run_glm52_colibri.sh status    # show what is installed/running
#   ./run_glm52_colibri.sh stop      # stop the colibri daemon
#
# Tested on Ubuntu 22.04 / Debian 12, macOS 14 (Apple Silicon), WSL2.
# ---------------------------------------------------------------------------

set -euo pipefail

# ===========================================================================
# Configuration  (override any of these with environment variables)
# ===========================================================================
COLIBRI_REPO="${COLIBRI_REPO:-https://github.com/JustVugg/colibri}"
COLIBRI_DIR="${COLIBRI_DIR:-$HOME/colibri}"
MODEL_ID="${MODEL_ID:-glm-5.2}"          # colibri model id, see README tables
BACKEND="${BACKEND:-auto}"              # auto | vulkan | cuda | cpu
MODEL_DIR="${MODEL_DIR:-}"               # optional: store model elsewhere
COLI_HOST="${COLI_HOST:-127.0.0.1}"
COLI_PORT="${COLI_PORT:-8000}"

# Approximate resource footprint for GLM 5.2 (from colibri README)
GLM52_DOWNLOAD_GB=429
GLM52_MIN_RAM_GB=16
GLM52_REC_RAM_GB=24

# Colors
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_CYAN=$'\033[36m'; C_YEL=$'\033[33m'
    C_GRN=$'\033[32m'; C_RED=$'\033[31m'; C_BOLD=$'\033[1m'
else
    C_RESET=""; C_CYAN=""; C_YEL=""; C_GRN=""; C_RED=""; C_BOLD=""
fi

log()  { printf '%s[%s]%s %s\n' "${C_CYAN}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
warn() { printf '%s[warn]%s %s\n'  "${C_YEL}"  "${C_RESET}" "$*"; }
ok()   { printf '%s[ok]%s %s\n'    "${C_GRN}"  "${C_RESET}" "$*"; }
err()  { printf '%s[err]%s %s\n'   "${C_RED}"  "${C_RESET}" "$*" >&2; }

# ===========================================================================
# Dependency detection
# ===========================================================================
detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *) echo "unknown" ;;
    esac
}

have() { command -v "$1" >/dev/null 2>&1; }

gb_free_in_dir() {
    # Print free space in GB for the directory that would hold $1
    local target="${1:-$HOME}"
    if have df; then
        df -BG --output=avail "$target" 2>/dev/null | tail -1 | tr -dc '0-9'
    fi
}

ensure_system_deps() {
    local os; os="$(detect_os)"
    log "Detected OS: $os"

    case "$os" in
        linux)
            if ! have git || ! have python3 || ! have gcc; then
                warn "Installing build deps with apt (needs sudo)..."
                sudo apt update
                sudo apt install -y git python3 build-essential cmake
            fi
            ;;
        macos)
            if ! xcode-select -p >/dev/null 2>&1; then
                warn "Installing Xcode command line tools..."
                xcode-select --install || true
            fi
            if ! have brew; then
                warn "Homebrew not found; please install from https://brew.sh"
            else
                brew list libomp >/dev/null 2>&1 || brew install libomp
                brew list git    >/dev/null 2>&1 || brew install git
            fi
            ;;
        windows)
            err "On Windows, use START-HERE.bat from the colibri repo instead."
            return 1
            ;;
        *)
            err "Unsupported OS: $(uname -s)"
            return 1
            ;;
    esac

    ok "System dependencies present"
}

# ===========================================================================
# Hardware pre-flight check (warn, do not fail, since colibri can run from RAM)
# ===========================================================================
check_hardware() {
    local ram_gb
    if [[ "$(detect_os)" == "macos" ]]; then
        ram_gb=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
    else
        ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
    fi
    log "RAM:        ${ram_gb} GB   (GLM 5.2 min ${GLM52_MIN_RAM_GB} GB, rec ${GLM52_REC_RAM_GB} GB)"
    (( ram_gb < GLM52_MIN_RAM_GB )) && \
        warn "Less than ${GLM52_MIN_RAM_GB} GB RAM detected - GLM 5.2 will be very slow or may not load."

    local free_gb; free_gb=$(gb_free_in_dir "$COLIBRI_DIR")
    if [[ -n "${free_gb:-}" ]]; then
        log "Free disk:  ${free_gb} GB in $COLIBRI_DIR  (GLM 5.2 download ~${GLM52_DOWNLOAD_GB} GB)"
        (( free_gb < GLM52_DOWNLOAD_GB )) && \
            warn "Not enough free disk for the ${GLM52_DOWNLOAD_GB} GB model download."
    fi

    # GPU hint
    if have nvidia-smi; then
        log "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ', ')"
    elif [[ "$(detect_os)" == "macos" ]]; then
        log "GPU:        Apple Silicon ($(sysctl -n machdep.cpu.brand_string))"
    else
        warn "No NVIDIA GPU detected - colibri will fall back to Vulkan/CPU."
    fi
}

# ===========================================================================
# colibri install / update
# ===========================================================================
clone_colibri() {
    if [[ -d "$COLIBRI_DIR/.git" ]]; then
        log "Colibri already cloned at $COLIBRI_DIR - pulling latest..."
        git -C "$COLIBRI_DIR" pull --ff-only
    else
        log "Cloning colibri into $COLIBRI_DIR ..."
        git clone --depth 1 "$COLIBRI_REPO" "$COLIBRI_DIR"
    fi
}

run_start_here() {
    [[ -d "$COLIBRI_DIR" ]] || clone_colibri
    local args=(--model "$MODEL_ID")
    [[ "$BACKEND" != "auto" ]] && args+=(--backend "$BACKEND")
    [[ -n "$MODEL_DIR" ]]     && args+=(--model-dir "$MODEL_DIR")

    log "Running start-here.sh ${args[*]}"
    ( cd "$COLIBRI_DIR" && bash ./start-here.sh "${args[@]}" )
}

# Path to the coli wrapper inside the cloned repo
coli_bin() { echo "$COLIBRI_DIR/coli"; }

# ===========================================================================
# Actions
# ===========================================================================
do_check() {
    ensure_system_deps
    check_hardware
}

do_install() {
    ensure_system_deps
    check_hardware
    clone_colibri
    run_start_here
    ok "GLM 5.2 installed. Try: $0 chat"
}

do_serve() {
    local coli; coli="$(coli_bin)"
    [[ -x "$coli" ]] || { err "colibri not installed - run '$0 install' first."; return 1; }
    log "Starting OpenAI-compatible API on http://${COLI_HOST}:${COLI_PORT}/v1"
    # coli serve reads COLI_HOST/COLI_PORT implicitly on most builds; if your
    # build does not, prefix with: COLI_HOST=$COLI_HOST COLI_PORT=$COLI_PORT
    exec "$coli" serve
}

do_chat() {
    local coli; coli="$(coli_bin)"
    [[ -x "$coli" ]] || { err "colibri not installed - run '$0 install' first."; return 1; }
    exec "$coli" chat
}

do_status() { "$(coli_bin)" status 2>/dev/null || err "colibri not installed"; }
do_stop()   { "$(coli_bin)" stop    2>/dev/null || err "nothing to stop"; ok "stopped"; }
do_logs()   { "$(coli_bin)" logs -n "${1:-50}" 2>/dev/null || err "no logs"; }

do_test() {
    log "Sending one test prompt to GLM 5.2 via the OpenAI-compatible endpoint..."
    curl -sS http://${COLI_HOST}:${COLI_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
            "model": "glm-5.2",
            "messages": [
                {"role": "system", "content": "You are GLM 5.2 running locally via colibri. Answer concisely."},
                {"role": "user",   "content": "Say hi in one sentence and tell me which model you are."}
            ],
            "stream": false,
            "temperature": 0.7
        }' | tee /tmp/glm52_colibri_test.json
    echo
    ok "Response saved to /tmp/glm52_colibri_test.json"
}

# ===========================================================================
# Usage
# ===========================================================================
usage() {
    cat <<EOF
${C_BOLD}run_glm52_colibri.sh${C_RESET} - run GLM 5.2 locally with colibri

${C_BOLD}USAGE${C_RESET}
    $0 <command> [args...]

${C_BOLD}COMMANDS${C_RESET}
    check      Verify OS deps + hardware (RAM, disk, GPU). No changes.
    install    Clone colibri and run start-here.sh --model glm-5.2
    serve      Start the OpenAI-compatible API on ${COLI_HOST}:${COLI_PORT}
    chat       Interactive chat REPL with GLM 5.2
    test       Curl the API with one test prompt
    status     Show what is installed and whether colibri is running
    stop       Stop the colibri daemon
    logs [N]   Show the last N log lines (default 50)
    help       Show this message

${C_BOLD}ENVIRONMENT (override as needed)${C_RESET}
    COLIBRI_REPO   ${COLIBRI_REPO}
    COLIBRI_DIR    ${COLIBRI_DIR}
    MODEL_ID       ${MODEL_ID}
    BACKEND        ${BACKEND}        # auto|vulkan|cuda|cpu
    MODEL_DIR      ${MODEL_DIR:-<unset>}
    COLI_HOST      ${COLI_HOST}
    COLI_PORT      ${COLI_PORT}

${C_BOLD}NOTES${C_RESET}
    * GLM 5.2 is a 744B MoE model (~${GLM52_DOWNLOAD_GB} GB download,
      ${GLM52_MIN_RAM_GB}-${GLM52_REC_RAM_GB} GB RAM). 'check' warns before you commit.
    * On Windows, prefer colibri's START-HERE.bat instead of this script.
    * The API is OpenAI-compatible: base url http://${COLI_HOST}:${COLI_PORT}/v1
EOF
}

# ===========================================================================
# Entry point
# ===========================================================================
case "${1:-help}" in
    check)   do_check ;;
    install) do_install ;;
    serve)   do_serve ;;
    chat)    do_chat ;;
    test)    do_test ;;
    status)  do_status ;;
    stop)    do_stop ;;
    logs)    do_logs "${2:-50}" ;;
    help|-h|--help) usage ;;
    *) err "Unknown command: $1"; usage; exit 1 ;;
esac
