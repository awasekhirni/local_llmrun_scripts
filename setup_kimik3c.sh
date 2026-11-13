#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# ==============================================================================
# Colors for output
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================================================${NC}"
echo -e "${GREEN}  Kimi K3 2.8T Model Local Setup Script (C99, 8GB RAM target)  ${NC}"
echo -e "${GREEN}=================================================================${NC}"

# ==============================================================================
# 1. Dependency Check
# ==============================================================================
echo -e "\n${BLUE}[1/6] Checking system dependencies...${NC}"
DEPS=("git" "make" "python3" "curl")
for cmd in "${DEPS[@]}"; do
    if ! command -v "$cmd" &> /dev/null; then
        echo -e "${RED}Error: '$cmd' is not installed. Please install it and try again.${NC}"
        exit 1
    fi
done

# Check for C compiler (GCC or Clang)
if command -v gcc &> /dev/null; then
    CC_VERSION=$(gcc -dumpversion)
    echo -e "${GREEN}✓ Found GCC v$CC_VERSION${NC}"
elif command -v clang &> /dev/null; then
    CC_VERSION=$(clang --version | head -n 1)
    echo -e "${GREEN}✓ Found Clang ($CC_VERSION)${NC}"
else
    echo -e "${RED}Error: No C compiler (GCC or Clang) found. Please install one.${NC}"
    exit 1
fi

# ==============================================================================
# 2. Storage Warning
# ==============================================================================
echo -e "\n${BLUE}[2/6] Checking storage requirements...${NC}"
echo -e "${YELLOW}WARNING: This model requires approximately 1.7 TB of free disk space.${NC}"
echo -e "${YELLOW}         (1.56 TB checkpoint + 109 GB packed trunk)${NC}"
df -h . | awk 'NR==2 {print "Current available storage on this drive: " $4 " (" $NF ")"}'
echo -e "${YELLOW}If you do not have ~1.7 TB free, the download will fail or fill your disk.${NC}"
read -p "Do you have sufficient storage and want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Setup aborted.${NC}"
    exit 1
fi

# ==============================================================================
# 3. Clone Repository
# ==============================================================================
echo -e "\n${BLUE}[3/6] Cloning repository...${NC}"
REPO_URL="https://github.com/FareedKhan-dev/kimi-k3-in-c.git"
REPO_DIR="kimi-k3-in-c"

if [ -d "$REPO_DIR" ]; then
    echo -e "${YELLOW}Directory '$REPO_DIR' already exists. Pulling latest changes...${NC}"
    cd "$REPO_DIR"
    git pull
else
    git clone "$REPO_URL"
    cd "$REPO_DIR"
fi

# ==============================================================================
# 4. System Check & Build
# ==============================================================================
echo -e "\n${BLUE}[4/6] Running system diagnostics and building...${NC}"
if [ -f "scripts/k3-doctor.sh" ]; then
    chmod +x scripts/k3-doctor.sh
    ./scripts/k3-doctor.sh
else
    echo -e "${YELLOW}Warning: k3-doctor.sh not found. Skipping hardware check.${NC}"
fi

echo -e "${BLUE}Compiling C code (this should take a few seconds)...${NC}"
make -j

echo -e "${BLUE}Running built-in tests to verify compilation...${NC}"
make test

# ==============================================================================
# 5. Download Checkpoint (Interactive)
# ==============================================================================
echo -e "\n${BLUE}[5/6] Model Download Preparation${NC}"
echo -e "${YELLOW}The model checkpoint is 1.56 TB. This will take hours depending on your bandwidth.${NC}"
echo -e "You need a Hugging Face account and a Read Token."
echo -e "Get one here: ${BLUE}https://huggingface.co/settings/tokens${NC}"

read -p "Enter your Hugging Face Token (starts with 'hf_'): " HF_TOKEN
if [[ -z "$HF_TOKEN" ]]; then
    echo -e "${RED}Error: Token cannot be empty.${NC}"
    exit 1
fi

export HF_TOKEN="$HF_TOKEN"
MODEL_DIR="$HOME/k3model"
TRUNK_DIR="$HOME/k3trunk"

echo -e "${BLUE}Starting download to $MODEL_DIR (resumable)...${NC}"
chmod +x scripts/download-model.sh
./scripts/download-model.sh "$MODEL_DIR"

# ==============================================================================
# 6. Pack Trunk & Final Instructions
# ==============================================================================
echo -e "\n${BLUE}[6/6] Packing the model trunk for optimized streaming...${NC}"
chmod +x scripts/pack-trunk.sh
./scripts/pack-trunk.sh "$MODEL_DIR" "$TRUNK_DIR"

echo -e "\n${GREEN}=================================================================${NC}"
echo -e "${GREEN}  Setup Complete! ${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo -e "You can now run the model locally. Example command:"
echo -e "${BLUE}./bin/k3 $MODEL_DIR --trunk $TRUNK_DIR --preset workstation \\"
echo -e "         --tok $MODEL_DIR --prompt \"The capital of France is\" --gen 8 --incremental${NC}"
echo -e "\nTo run it again in the future, simply navigate to the '$REPO_DIR' folder and execute the above command."
echo -e "${YELLOW}Note: Ensure you have fast local disk (NVMe recommended) for acceptable token generation speeds.${NC}"
