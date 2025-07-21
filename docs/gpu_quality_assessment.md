# GPU-Accelerated Quality Assessment

QuarryCore now supports GPU acceleration for quality assessment, providing significant performance improvements when CUDA-capable hardware is available.

## Features

- **Automatic GPU Detection**: Automatically detects and uses CUDA when available
- **Transparent Fallback**: Seamlessly falls back to CPU when CUDA is not available
- **Configurable Device Selection**: Choose between `auto`, `cpu`, or `cuda` modes
- **Performance Monitoring**: Built-in metrics for tracking scorer latency and errors

## Configuration

Configure the device in your `config.yaml`:

```yaml
quality:
  device: auto  # Options: "auto", "cpu", "cuda"
  min_content_length: 50
  max_content_length: 50000
```

Or use environment variables:

```bash
export QUARRY_QUALITY_DEVICE=cuda
```

## Device Options

- **`auto`** (default): Automatically selects CUDA if available, otherwise CPU
- **`cpu`**: Forces CPU execution even if CUDA is available
- **`cuda`**: Uses CUDA if available, falls back to CPU with a warning

## Performance Metrics

The GPU-accelerated scorer tracks performance metrics:

- `quarrycore_quality_scorer_latency_seconds`: Latency histogram by scorer type
- `quarrycore_quality_scorer_errors_total`: Error counter by scorer type
- `quarrycore_quality_reject_total`: Documents rejected due to low quality

## Requirements

### CPU Mode
- No additional requirements
- Works on all systems

### GPU Mode
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- PyTorch with CUDA support
- Sufficient GPU memory (typically < 1GB for small models)

## Performance Expectations

When CUDA is available:
- **Median latency**: ≤ 25ms for 1KB English text
- **Memory usage**: < 300MB RAM delta after model initialization
- **GPU memory**: < 1GB for sentence-transformers/all-MiniLM-L6-v2

## Testing

### Unit Tests (No GPU Required)
```bash
export QUARRY_QUALITY_DEVICE=cpu
pytest tests/unit/quality/test_transformer_gpu.py -v
```

### Performance Tests (GPU Required)
```bash
export QUARRY_QUALITY_DEVICE=cuda
pytest tests/performance/test_quality_gpu_perf.py -v -m requires_cuda
```

## Troubleshooting

### CUDA Not Detected
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of Memory Errors
- Reduce batch size in configuration
- Use a smaller model
- Clear GPU cache periodically

### Performance Issues
- Ensure GPU is not being used by other processes
- Check GPU utilization with `nvidia-smi`
- Consider using CPU mode for small workloads where GPU overhead isn't justified 