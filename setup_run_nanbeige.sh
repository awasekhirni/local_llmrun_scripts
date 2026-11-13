#!/bin/bash

# ==============================================================================
# Bash script to run Nanbeige4.1-3B model from Hugging Face
# Model: https://huggingface.co/Nanbeige/Nanbeige4.1-3B
# chmod +x run_nanbeige.sh
#   ./run_nanbeige.sh
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

MODEL_ID="Nanbeige/Nanbeige4.1-3B"
VENV_DIR="nanbeige_venv"
SCRIPT_NAME="inference.py"

echo "=================================================="
echo "  Setting up environment for Nanbeige4.1-3B"
echo "=================================================="

# 1. Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Please install Python 3.9+ and try again."
    exit 1
fi

# 2. Create and activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment in ./$VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Install dependencies
echo "  Installing required Python packages (this may take a few minutes)..."
pip install --upgrade pip
# Installing torch without explicit CUDA index ensures cross-platform compatibility
# (Linux CUDA, Mac MPS, or CPU fallback)
pip install torch transformers accelerate

# 4. Create the Python inference script
echo " Creating inference script: $SCRIPT_NAME"
cat << 'EOF' > "$SCRIPT_NAME"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Nanbeige/Nanbeige4.1-3B"

def main():
    print(" Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=False,
        trust_remote_code=True
    )

    print(" Loading model... (this may take a moment depending on your internet speed)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    # Example chat scenario (You can change this prompt)
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

    # Decode only the generated part (excluding the prompt)
    resp = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

    print("\n" + "="*60)
    print("  MODEL RESPONSE:")
    print("="*60)
    print(resp)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
EOF

# 5. Run the inference script
echo "=================================================="
echo "   Running Nanbeige4.1-3B Inference"
echo "=================================================="
python3 "$SCRIPT_NAME"

echo "=================================================="
echo "  Inference complete!"
echo "  To run again later without reinstalling, use:"
echo "    source $VENV_DIR/bin/activate"
echo "    python3 $SCRIPT_NAME"
echo "=================================================="
