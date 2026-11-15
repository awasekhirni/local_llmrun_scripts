#!/bin/bash

# ==============================================================================
# Script to install and run an 8B model locally in strict FP16 precision
# from openai import OpenAI
#
# client = OpenAI(base_url="http://localhost:8000/v1", api_key="empty")
#
#response = client.chat.completions.create(
#    model="Qwen/Qwen2.5-8B",
#    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}]
#)
#print(response.choices[0].message.content)
# ==============================================================================

set -e

# --- Configuration ---
# Change this to "meta-llama/Llama-3.2-8B" if you prefer Llama
MODEL_NAME="Qwen/Qwen2.5-8B" 
VENV_DIR="llm_fp16_env"
PORT=8000

echo "========================================="
echo " Setting up local FP16 LLM Environment"
echo "========================================="
echo "NOTE: FP16 8B requires ~16-18GB VRAM. Ensure you have a compatible NVIDIA GPU."

# 1. Create and activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating Python virtual environment..."
    python3 -m venv $VENV_DIR
fi

echo "==> Activating virtual environment..."
source $VENV_DIR/bin/activate

# 2. Install vLLM (Optimized for FP16 inference)
echo "==> Installing vLLM and dependencies..."
pip install --upgrade pip
pip install vllm

# 3. Start the vLLM OpenAI-compatible API server in the background
echo "==> Starting $MODEL_NAME in FP16 precision..."
echo "    (If this is your first run, it will download the ~16GB model weights)"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --dtype float16 \
    --port $PORT \
    --max-model-len 4096 &

SERVER_PID=$!

# 4. Wait for the server to become healthy
echo "==> Waiting for model to load into VRAM..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health > /dev/null; then
        echo "==> Server is ready!"
        break
    fi
    sleep 2
done

# 5. Test the model with a sample prompt
echo "==> Sending a test prompt..."
curl http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$MODEL_NAME'",
        "messages": [
            {"role": "user", "content": "Hello! Can you briefly introduce yourself?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'

echo -e "\n\n========================================="
echo " Setup Complete!"
echo "========================================="
echo "==> API Server is running at: http://localhost:$PORT"
echo "==> Server PID: $SERVER_PID"
echo "==> To stop the server, press Ctrl+C or run: kill $SERVER_PID"
echo "========================================="

# Keep the script alive so the server keeps running in the foreground
wait $SERVER_PID
