# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TIGER (Time-frequency Interleaved Gain Extraction and Reconstruction) is a lightweight speech separation model. It separates mixed audio into individual speaker tracks using frequency band-split and interleaved time-frequency modeling.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Inference with pre-trained models
python inference_speech.py --audio_path test/mix.wav --output_dir separated_audio
python inference_dnr.py --audio_path test/test_mixture_466.wav  # Dialog/Music/Effects separation

# Training
python audio_train.py --conf_dir configs/tiger-large.yml

# Evaluation
python audio_test.py --conf_dir configs/tiger-large.yml

# Compute MACs and parameters
python evaluated_mac_params.py
```

## Architecture

The codebase uses PyTorch Lightning for training with the `look2hear` package structure:

```
look2hear/
├── models/       # TIGER and TIGERDNR model implementations
├── datas/        # DataModules for EchoSet, Libri2Mix, LRS2
├── losses/       # PITLossWrapper with SNR/SI-SDR losses
├── metrics/      # MetricsTracker for evaluation
├── layers/       # Building blocks (STFT, normalizations, activations)
├── system/       # AudioLightningModule, optimizers, schedulers
└── utils/        # Parser utilities, separator helpers
```

Key model components in `look2hear/models/tiger.py`:
- `TIGER`: Main model class extending `BaseModel`, handles STFT/iSTFT and frequency band splitting
- `Recurrent`: Core separator with iterative frequency-time processing
- `UConvBlock`: Multi-scale selective attention (MSA) module with dilated convolutions
- `MultiHeadSelfAttention2D`: Full-frequency-frame attention (F³A) module

## Configuration

Training configs are YAML files in `configs/` with sections:
- `audionet`: Model architecture (TIGER, channels, blocks, window size)
- `datamodule`: Dataset paths, sample rate, batch size
- `training`: GPUs, epochs, early stopping
- `optimizer`/`scheduler`: Adam with ReduceLROnPlateau
- `loss`: PITLossWrapper with train/val loss types

## Data Preprocessing

Scripts in `DataPreProcess/` prepare datasets:
- `process_echoset.py` - EchoSet (noise + reverb)
- `process_librimix.py` - Libri2Mix
- `preprocess_lrs2_audio.py` - LRS2-2Mix

## Pre-trained Models

Models auto-download from HuggingFace:
- `JusperLee/TIGER-speech` - 2-speaker separation
- `JusperLee/TIGER-speech-small`, `JusperLee/TIGER-speech-tiny` - Smaller variants
