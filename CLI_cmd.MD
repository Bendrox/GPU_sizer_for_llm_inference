# Monitoring the GPU

A few handy commands to check that a GPU is detected and to track its load during inference.

## Checking the setup

General state of the card (driver, VRAM, running processes):

```bash
nvidia-smi
```

Confirm that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True NVIDIA GeForce RTX 4090` (or your card's name). A `False` usually points to a driver issue or a PyTorch build without CUDA support.

## Tracking load in real time

Refresh `nvidia-smi` every 0.5 s:

```bash
watch -n 0.5 nvidia-smi
```

Keep only the metrics that matter, in CSV, one line per second — convenient for logging to a file:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1
```

An interactive, `htop`-style view, easier to read than `nvidia-smi` for watching things evolve:

```bash
nvtop
```

## Notes

- `watch` and `nvtop` are Linux tools. On Windows, rely on `nvidia-smi`'s built-in loop (`-l 1`) or the Task Manager.
- `nvtop` installs via your package manager (`apt install nvtop`, `brew install nvtop`...).
- Redirecting the CSV output to a file lets you plot power draw afterwards: `... --format=csv -l 1 > gpu.csv`.
