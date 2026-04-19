# Qwen-Image-Edit Speed Research Report

## Executive Summary
**Qwen-Image-Edit is SLOWER than the current models used in the codebase.** The codebase uses FLUX.2-klein-4B and Z-Image Turbo, both of which offer sub-second inference (under 0.5s). Qwen-Image-Edit standard inference takes 2.1-4.7s, and even its Lightning variant (4-step) is likely slower than FLUX.2-klein's native 4-step distilled architecture.

## Codebase Current Models

The manga-to-realistic codebase uses these models:

1. **FLUX.2-klein-4B (Int8)** - FAST_FLUX_MODEL_CHOICE (default)
2. **FLUX.2-klein-4B (4bit SDNQ - Low VRAM)** - LOW_VRAM_FLUX_MODEL_CHOICE
3. **Z-Image Turbo (Int8 - 8GB Safe)** - Alternative option

## Model Specifications

### Qwen-Image-Edit
- **Parameters**: 20 billion
- **Architecture**: Multi-modal Diffusion Transformer (MMDiT)
- **License**: Apache 2.0 (Commercial-friendly)
- **Input Resolution**: Up to 1024×1024 pixels
- **Text Support**: Bilingual (English and Chinese)
- **Memory Requirements**: 24GB+ VRAM (quantization options available)

## Performance Benchmarks

### Codebase Models Performance

#### FLUX.2-klein-4B (Current Default)
- **Inference Time**: Sub-second (<0.5s) on modern hardware
- **Steps**: 4-step distilled (native architecture)
- **VRAM**: ~13GB (RTX 3090/4070 and above)
- **Quantized Versions**:
  - Int8: Default in codebase
  - 4bit SDNQ: Low VRAM option
  - FP8: Up to 1.6x faster, 40% less VRAM
  - NVFP4: Up to 2.7x faster, 55% less VRAM
- **Quality**: Matches or exceeds models 5x its size
- **License**: Apache 2.0 (4B models)

#### Z-Image Turbo (Alternative)
- **Inference Time**: Sub-second latency on enterprise-grade H800 GPUs
- **Steps**: 8 NFEs (Number of Function Evaluations)
- **VRAM**: Fits in 16GB consumer devices
- **Quality**: State-of-the-art results among open-source models
- **Text Rendering**: Excellent bilingual (Chinese/English) text rendering

### Qwen-Image-Edit Performance

#### Standard Inference (50 steps)
| GPU | Time | VRAM Usage | Quality Score |
|-----|------|------------|---------------|
| RTX 4090 | 3.2s | 22.1 GB | 95.2 |
| RTX 3090 | 4.7s | 23.8 GB | 95.2 |
| RTX 3090 (4-bit) | 5.8s | 12.4 GB | 91.7 |
| A100 | 2.1s | 21.3 GB | 95.2 |

#### Qwen-Image-Lightning (4-step inference)
- **Speedup**: Up to 25x faster than standard 40-step inference
- **Performance**: Roughly 10 times faster than standard inference
- **Estimated Time**: ~0.3-0.5s (estimated from 3.2s / 10x)
- **Quality**: Maintains high visual fidelity with minimal quality loss
- **Note**: Lightning is a LoRA add-on, not native distilled architecture

## Direct Speed Comparison

| Model | Steps | Inference Time | VRAM | Notes |
|-------|-------|----------------|------|-------|
| **FLUX.2-klein-4B** | 4 (native) | <0.5s | ~13GB | **Current default** |
| **FLUX.2-klein-4B (NVFP4)** | 4 (native) | ~0.2s | ~6GB | 2.7x faster than base |
| **Z-Image Turbo** | 8 NFEs | <1s | 16GB | Alternative option |
| **Qwen-Image-Lightning** | 4 (LoRA) | ~0.3-0.5s | 24GB+ | LoRA add-on |
| **Qwen-Image-Edit** | 50 (standard) | 2.1-4.7s | 24GB+ | Not competitive |

## Key Findings

### Speed Comparison
1. **FLUX.2-klein-4B is fastest**: Native 4-step distilled architecture, sub-second inference
2. **Qwen-Image-Lightning is competitive**: Similar 4-step approach but as LoRA add-on
3. **Standard Qwen-Image-Edit is much slower**: 50 steps, 4-10x slower than codebase models
4. **Z-Image Turbo is fast**: 8 NFEs, sub-second on modern hardware

### Quality Comparison
- **FLUX.2-klein**: Official benchmarks state it "matches or exceeds Qwen's quality at a fraction of the latency and VRAM"
- **Qwen-Image-Edit**: Excellent text rendering (especially Chinese), but slower
- **Z-Image Turbo**: State-of-the-art among open-source models, excellent bilingual text

### Memory Efficiency
1. **FLUX.2-klein-4B**: ~13GB VRAM (fits RTX 3090/4070)
2. **FLUX.2-klein-4B (NVFP4)**: ~6GB VRAM (55% reduction)
3. **Z-Image Turbo**: 16GB VRAM
4. **Qwen-Image-Edit**: 24GB+ VRAM (highest requirement)
5. **Qwen-Image-Edit (4-bit)**: 12.4GB VRAM (with quality loss to 91.7)

## Recommendations

### For the Manga-to-Realistic Project

**DO NOT switch to Qwen-Image-Edit.** The current codebase models are superior:

1. **Keep FLUX.2-klein-4B as default**: It's faster (<0.5s vs 2.1-4.7s), uses less VRAM (~13GB vs 24GB+), and matches or exceeds Qwen's quality according to official benchmarks.

2. **Consider FLUX.2-klein-4B NVFP4 for low VRAM**: If users have limited VRAM, the NVFP4 variant offers 2.7x speedup and 55% VRAM reduction (~6GB).

3. **Z-Image Turbo remains a good alternative**: Sub-second inference, excellent bilingual text rendering, fits in 16GB VRAM.

### When Qwen-Image-Edit Might Be Useful

Qwen-Image-Edit could be considered only if:
- You need specialized Chinese text editing capabilities (94.1 benchmark score vs FLUX's lower scores)
- You have 24GB+ VRAM and can accept slower inference
- You need specific image editing features that Qwen offers but FLUX doesn't

However, given that FLUX.2-klein officially "matches or exceeds Qwen's quality at a fraction of the latency and VRAM," there's little reason to switch.

## Conclusion

**Qwen-Image-Edit is NOT faster than the current models in the codebase.** The codebase uses FLUX.2-klein-4B and Z-Image Turbo, both of which offer sub-second inference (<0.5s and <1s respectively). Qwen-Image-Edit standard inference takes 2.1-4.7s, and even its Lightning variant is likely comparable to or slower than FLUX.2-klein's native 4-step distilled architecture.

**Key advantages of current codebase models:**
- **FLUX.2-klein-4B**: Sub-second inference, ~13GB VRAM, Apache 2.0 license, matches/exceeds Qwen quality
- **Z-Image Turbo**: Sub-second inference, 16GB VRAM, excellent bilingual text rendering
- **Both models**: Already integrated, tested, and optimized for the codebase

**Recommendation**: Stick with current models (FLUX.2-klein-4B and Z-Image Turbo). They are faster, more memory-efficient, and offer comparable or better quality than Qwen-Image-Edit.
