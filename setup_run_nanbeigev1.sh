#!/bin/bash

# ==============================================================================
# Smart Bash script to run Nanbeige4.1-3B with dynamic 50% RAM allocation
# Detects total system RAM, calculates 50%, and constrains PyTorch to that limit.
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

MODEL_ID="Nanbeige/Nanbeige4.1-3B"
VENV_DIR="nanbeige_venv"
SCRIPT_NAME="inference_smart.py"

echo "=================================================="
echo " Detecting System Memory..."
echo "=================================================="

# 1. Detect Total RAM based on OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux: Get MemTotal in kB, convert to GB
    TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_RAM_GB=$(( TOTAL_RAM_KB / 1024 / 1024 ))
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: Get hw.memsize in bytes, convert to GB
    TOTAL_RAM_BYTES=$(sysctl -n hw.memsize)
    TOTAL_RAM_GB=$(( TOTAL_RAM_BYTES / 1024 / 1024 / 1024 ))
else
    echo "  Unsupported OS. Defaulting to 32GB limit."
    TOTAL_RAM_GB=64
fi

# 2. Calculate 50% of Total RAM (Minimum 8GB to ensure model can load)
TARGET_RAM_GB=$(( TOTAL_RAM_GB / 2 ))
if [ "$TARGET_RAM_GB" -lt 8 ]; then
    TARGET_RAM_GB=8
fi

echo " Total System RAM: ${TOTAL_RAM_GB} GB"
echo " Allocated for AI Workload (50%): ${TARGET_RAM_GB} GB"
echo " Reserved for OS/Other (50%): $(( TOTAL_RAM_GB - TARGET_RAM_GB )) GB"
echo "=================================================="

# 3. Create and activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo " Creating virtual environment in ./$VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 4. Install dependencies
echo "  Installing required Python packages..."
pip install --upgrade pip
pip install torch transformers accelerate

# 5. Export the calculated memory limit for the Python script
export MAX_MEMORY_GB="$TARGET_RAM_GB"

# 6. Create the Python inference script
echo "📝 Creating dynamic inference script: $SCRIPT_NAME"
cat << 'EOF' > "$SCRIPT_NAME"
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Nanbeige/Nanbeige4.1-3B"

def main():
    # Read the 50% RAM limit passed from the bash script
    max_mem_gb = int(os.environ.get("MAX_MEMORY_GB", 32))
    max_mem_str = f"{max_mem_gb}GiB"

    print(f"  Configured maximum memory limit: {max_mem_str}")

    # Dynamically set max_memory based on available hardware accelerators
    if torch.cuda.is_available():
        print(" Detected NVIDIA GPU. Allocating memory to CUDA.")
        max_memory = {0: max_mem_str, "cpu": max_mem_str}
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(" Detected Apple Silicon. Allocating memory to MPS.")
        max_memory = {"mps": max_mem_str, "cpu": max_mem_str}
    else:
        print(" No GPU detected. Running on CPU with memory limit.")
        max_memory = {"cpu": max_mem_str}

    print(" Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=False,
        trust_remote_code=True
    )

    print(f" Loading model with max_memory={max_memory}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        max_memory=max_memory,  # <-- Enforces the 50% RAM limit
        trust_remote_code=True
    )

    # Example chat scenario
    messages = [
        {"role": "user", "content": "Which number is bigger, 9.11 or 9.8? Explain your reasoning step-by-step."}
    ]

    print(" Applying chat template...")
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    print(" Generating response...\n")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            eos_token_id=166101  # Specific EOS token for Nanbeige models
        )

    resp = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

    print("\n" + "="*60)
    print(" MODEL RESPONSE:")
    print("="*60)
    print(resp)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
EOF

# 7. Run the inference script
echo "=================================================="
echo "  Running Nanbeige4.1-3B with ${TARGET_RAM_GB}GB Memory Limit"
echo "=================================================="
python3 "$SCRIPT_NAME"

echo "=================================================="
echo " Inference complete!"
echo " To run again later without reinstalling, use:"
echo "    source $VENV_DIR/bin/activate"
echo "    export MAX_MEMORY_GB=$TARGET_RAM_GB"
echo "    python3 $SCRIPT_NAME"
echo "=================================================="
