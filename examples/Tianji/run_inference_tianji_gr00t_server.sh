#!/usr/bin/env bash
# Tianji GR00T inference server. Edit the configuration block below, then run this script directly.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# =============================================================================
# 推理参数（直接修改这里）
# =============================================================================
RUN_DIR="/home/standard/workspace/tianji_demo/checkpoints/Isaac-GR00T/tianji_20d_run_001"
# 10000 | 15000 | 20000；20000 使用 RUN_DIR 根目录。
CHECKPOINT_STEP="10000"
COSMOS_MODEL_NAME="nvidia/Cosmos-Reason2-2B"
# 保留现有缓存位置，但允许 Hugging Face 联网检查和下载缺失文件。
HF_HOME="/home/standard/checkpoints/cache/huggingface"
HF_ENDPOINT="https://huggingface.co"
CUDA_VISIBLE_DEVICES="0"
DEVICE="cuda:0"
HOST="0.0.0.0"
PORT="5555"
PYTHON="$REPO_ROOT/.venv/bin/python"

case "$CHECKPOINT_STEP" in
    10000|15000)
        MODEL_PATH="$RUN_DIR/checkpoint-$CHECKPOINT_STEP"
        ;;
    20000)
        MODEL_PATH="$RUN_DIR"
        ;;
    *)
        echo "ERROR: CHECKPOINT_STEP must be 10000, 15000, or 20000; got $CHECKPOINT_STEP" >&2
        exit 2
        ;;
esac

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file not found: $1" >&2
        exit 1
    fi
}

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: checkpoint directory not found: $MODEL_PATH" >&2
    exit 1
fi
require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"

PROCESSOR_DIR="$MODEL_PATH"
if [[ -d "$MODEL_PATH/processor" && ! -f "$MODEL_PATH/processor_config.json" ]]; then
    PROCESSOR_DIR="$MODEL_PATH/processor"
fi
require_file "$PROCESSOR_DIR/processor_config.json"
require_file "$PROCESSOR_DIR/statistics.json"
require_file "$PROCESSOR_DIR/embodiment_id.json"

"$PYTHON" - "$MODEL_PATH" <<'PY'
import json
from pathlib import Path
import sys

model_path = Path(sys.argv[1])
with open(model_path / "model.safetensors.index.json") as f:
    index = json.load(f)
missing = [
    str(model_path / shard)
    for shard in sorted(set(index["weight_map"].values()))
    if not (model_path / shard).is_file()
]
if missing:
    raise SystemExit("ERROR: missing model shard(s):\n  " + "\n  ".join(missing))
PY

export HF_HOME
export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

"$PYTHON" - "$COSMOS_MODEL_NAME" <<'PY'
from huggingface_hub import HfApi, get_token
import sys

model_name = sys.argv[1]
token = get_token()
if token is None:
    raise SystemExit(
        "ERROR: no Hugging Face token found under the configured HF_HOME. "
        "Run `hf auth login` with the same HF_HOME first."
    )
try:
    HfApi().model_info(model_name, token=token)
except Exception as exc:
    raise SystemExit(f"ERROR: Hugging Face access check failed for {model_name}: {exc}") from exc
print(f"Hugging Face access OK: {model_name}")
PY

echo "Starting Tianji GR00T inference server"
echo "  checkpoint step: $CHECKPOINT_STEP"
echo "  model path:      $MODEL_PATH"
echo "  Cosmos source:   $COSMOS_MODEL_NAME"
echo "  HF endpoint:     $HF_ENDPOINT"
echo "  HF cache:        $HF_HOME"
echo "  CUDA devices:    $CUDA_VISIBLE_DEVICES"
echo "  device:          $DEVICE"
echo "  listen:          $HOST:$PORT"

cd "$REPO_ROOT"
exec "$PYTHON" gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --model-name "$COSMOS_MODEL_NAME" \
    --embodiment-tag NEW_EMBODIMENT \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT"
