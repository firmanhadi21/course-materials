# Satellite Super-Resolution Workshop

Code for the [Satellite Super-Resolution Workshop](https://firmanhadi21.github.io/blog/materials/20260326_Super-resolution_workshop.html).

## Contents

### Data Pipeline
- `dsen2_harmonize.py` - Sharpen Sentinel-2 20m/60m bands to 10m using DSen2
- `crop_pairs.py` - Crop S2 scenes to match PlanetScope extent
- `preprocess.py` - Co-register images and extract aligned patch pairs
- `dataset.py` - Patch loader with normalization and augmentation

### Model & Training
- `s2ps_archs.py` - Network architectures (EDSR, SwinIR, BandAdapterNet)
- `s2ps_losses.py` - Custom loss functions (SAM, ERGAS)
- `s2ps_model.py` - Training logic (forward pass, loss, validation)
- `s2ps_dataset.py` - Dataset loader registration
- `s2ps_metrics.py` - Quality metrics (PSNR, SSIM, SAM, ERGAS)
- `launch.py` - Entry point for BasicSR framework
- `train_example.py` - Standalone minimal training script

### Inference
- `inference.py` - Apply trained model to full scenes with tiled processing

### Configs
- `swinir_semarang.yml` - SwinIR training config
- `esrgan_semarang.yml` - ESRGAN training config
- `bandadapter_esrgan_s2ps.yml` - BandAdapter-ESRGAN config

## Requirements

See `requirements.txt`. Key dependencies: PyTorch, BasicSR, NumPy, GDAL.
