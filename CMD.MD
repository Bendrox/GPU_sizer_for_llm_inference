## Commandes :
- nvidia-smi 
- python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
- watch -n 0.5 nvidia-smi
- nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1
- nvtop 