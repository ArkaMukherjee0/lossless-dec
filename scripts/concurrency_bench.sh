#!/usr/bin/env bash
# Phase D: throughput vs concurrency with continuous arrivals (runs inside the container).
# Usage: concurrency_bench.sh <model> <tag> <spec-json|""> <c1,c2,...>
set -euo pipefail
MODEL=$1; TAG=$2; SPEC=$3; CONCS=$4; TP=${TP:-2}; EXTRA=${EXTRA:-}
OUT=/workspace/results/d/${MODEL##*/}; mkdir -p "$OUT"
PORT=8000

vllm serve "$MODEL" --tensor-parallel-size $TP $EXTRA --port $PORT --max-model-len 8192 \
  --gpu-memory-utilization ${GPU_MEM:-0.85} --limit-mm-per-prompt '{"image":0,"video":0}' \
  ${SPEC:+--speculative-config "$SPEC"} > "$OUT/${TAG}_server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf localhost:$PORT/health >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -30 "$OUT/${TAG}_server.log"; exit 1; }
  sleep 5
done

for C in ${CONCS//,/ }; do
  N=$(( C * 4 > 64 ? C * 4 : 64 ))
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --model "$MODEL" --port $PORT \
    --dataset-name spec_bench --dataset-path /workspace/data/spec_bench.jsonl --spec-bench-output-len 512 \
    --num-prompts $N --max-concurrency $C --num-warmups $C --temperature 0 --seed 0 \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
    --save-result --result-dir "$OUT" --result-filename "${TAG}_c${C}.json" > "$OUT/${TAG}_c${C}.log" 2>&1
  python3 -c "import json; r=json.load(open('$OUT/${TAG}_c${C}.json')); print('c=$C', round(r['output_throughput'], 1), 'tok/s, acceptance length', r.get('spec_decode_acceptance_length'))"
done
