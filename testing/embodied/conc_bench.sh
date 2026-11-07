#!/bin/bash
GGUF=/work/kubuds/hermes/models/qwen3_8b_gguf/Qwen3-8B-Q4_K_M.gguf
PY=/work/kubuds/hermes/vllm_env/bin/python
ACT=/work/kubuds/hermes/vla_test/models/act/models/onnx/act-int8/act.q.onnx
llama-bench -m "$GGUF" -p 0 -n 64 -t 4 -r 1 2>/dev/null | grep tg64 > /tmp/llama_conc.txt &
LL=$!
sleep 12
vmstat 2 40 > /tmp/vm_conc.txt &
VM=$!
$PY - "$ACT" <<'EOF'
import sys, time, numpy as np, onnxruntime as ort
so = ort.SessionOptions(); so.log_severity_level = 3
s = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
img = np.zeros((1, 2, 3, 480, 640), np.float32); st = np.zeros((1, 6), np.float32)
s.run(None, {"images": img, "state": st})
t0 = time.time()
for _ in range(3):
    s.run(None, {"images": img, "state": st})
print(f"ACT int8 concurrent: {(time.time()-t0)/3:.2f} s/iter")
EOF
wait $LL; kill $VM 2>/dev/null
echo -n "llama concurrent: "; cat /tmp/llama_conc.txt
awk 'NR>2 {r+=$13; u+=$14; s+=$15; n++} END {printf "vmstat avg: us=%d%% sy=%d%% id=%d%% (n=%d)\n", u/n, s/n, r/n, n}' /tmp/vm_conc.txt
