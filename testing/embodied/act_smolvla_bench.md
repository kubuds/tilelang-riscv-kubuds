# ACT / smolVLA 实测原始记录（2026-11-07, K3 CPU EP）

ACT fp32: load 0.90s, infer 3.82s/chunk(100 step)
ACT int8: load 0.44s, infer 2.77s/chunk -> 0.36 Hz
smolVLA: load 6.69s, vision+connector 23.13s, prefill_lm 2.98s, denoise x10 7.55s (755ms/step), total 33.7s/chunk
SpaceMIT EP: RuntimeException tcm buffer acquire failed for core id 0; /dev/tcm_sync_mem missing
concurrent: ACT 2.94s(+6pct), llama 8B Q4_K_M 3.97 tok/s (solo 3.92 @t4); vmstat us1 sy78 id15
ROS2: pub 256B 6112 msg/s, 4KB 921 msg/s (3.8 MB/s)
