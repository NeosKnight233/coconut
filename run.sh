export NCCL_TIMEOUT=1800000
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=WARN          # Options: TRACE, INFO, WARN, ERROR
export NCCL_DEBUG_SUBSYS=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF  # Options: OFF, INFO, DETAIL
export TRANSFORMERS_VERBOSITY=warning
torchrun --nproc_per_node=4 --master_port=29500 run.py args/gsm_coconut_llama3.2_3b.yaml