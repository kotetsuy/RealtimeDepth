"""Shared device validation for the server and inference benchmark."""
import os

# Native gfx1151 kernels must see the real GPU architecture, before HIP loads.
os.environ.pop('HSA_OVERRIDE_GFX_VERSION', None)

import torch


def configure_runtime(config):
    device = torch.device(config.get('device', 'cuda'))
    precision = config.get('precision', 'fp16')
    if precision not in ('fp16', 'fp32'):
        raise ValueError('runtime.precision must be fp16 or fp32')
    if device.type == 'cuda':
        if not torch.version.hip:
            raise RuntimeError('ROCm PyTorch is required. Run bash setup_rocm10.sh')
        if not torch.cuda.is_available():
            raise RuntimeError(
                'No HIP GPU available. Check /dev/kfd, /dev/dri and render/video '
                'group permissions; run bash setup_rocm10.sh for ROCm 10 wheels.')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        print(f'Device: {props.name} ({props.gcnArchName})', flush=True)
    print(f'PyTorch: {torch.__version__}; HIP: {torch.version.hip}', flush=True)
    return str(device), torch.float16 if precision == 'fp16' else torch.float32
