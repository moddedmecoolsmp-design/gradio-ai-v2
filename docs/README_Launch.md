# Ultra Fast Image Gen - Windows Launch Scripts

## Overview

This project now uses two separate batch scripts for optimal user experience:

- **`Verify.bat`** - Comprehensive system verification (run first)
- **`Launch.bat`** - Fast application launcher (run after verification)

## Quick Start

### First Time Setup
```batch
# 1. Install dependencies
Install.bat

# 2. Verify your setup
Verify.bat

# 3. Launch the application
Launch.bat
```

### Regular Usage
```batch
# Just launch (assumes verification passed)
Launch.bat
```

## Script Details

### Verify.bat - System Verification
**Purpose**: Comprehensive pre-flight checks before launching

**What it validates**:
- âœ… Windows version compatibility
- âœ… Virtual environment integrity
- âœ… Python version (3.10+ required)
- âœ… PyTorch installation and CUDA availability
- âœ… Critical packages (Gradio, Transformers, Diffusers)
- âœ… Optional components (git-lfs, qwen-tts)
- âœ… GPU/CUDA setup
- âœ… Directory structure
- âœ… Environment configuration

**When to run**: First time, after system changes, or if Launch.bat fails

### Launch.bat - Application Launcher
**Purpose**: Fast, streamlined application startup

**What it does**:
- âœ… Basic environment checks
- âœ… Virtual environment activation
- âœ… Environment variable configuration
- âœ… Port cleanup and process management
- âœ… Browser auto-launch
- âœ… Application startup

**Prerequisites**: Requires successful verification (run Verify.bat first)

## Workflow Comparison

### Before (Single Script)
```
Launch.bat â†’ [8-step validation] â†’ [launch]
Time: ~2-3 minutes (including validation)
```

### After (Separated Scripts)
```
Setup: Install.bat â†’ Verify.bat â†’ Launch.bat
Daily: Launch.bat only
Time: ~30 seconds for launch
```

## Error Handling

### If Launch.bat fails
1. Run `Verify.bat` to diagnose issues
2. Follow troubleshooting steps provided
3. Re-run `Install.bat` if dependencies are missing
4. Run `Install.bat --repair` if dependency profile is stale or invalid

### Common Issues
- **"Virtual environment not found"**: Run `Install.bat`
- **"Application exited with error"**: Check Verify.bat output
- **Unicode errors**: Windows console encoding issue (cosmetic only)

## File Structure

```
manga-to-realistic/
â”œâ”€â”€ Install.bat          # Initial setup
â”œâ”€â”€ Verify.bat           # System verification
â”œâ”€â”€ Launch.bat           # Application launcher
â”œâ”€â”€ Launch_Comprehensive.bat  # Full verification + launch (backup)
â””â”€â”€ app.py               # Main application
```

## Performance Benefits

- **Faster daily launches**: 30 seconds vs 2-3 minutes
- **Clear separation of concerns**: Verification vs launching
- **Better error diagnosis**: Dedicated verification step
- **Reduced complexity**: Simpler, more reliable launch script

## Migration Notes

- **Existing users**: No action required, both scripts work
- **New installations**: Use the new workflow for best experience
- **Old comprehensive script**: Preserved as `Launch_Comprehensive.bat`

---

**Recommendation**: Always run `Verify.bat` after `Install.bat` and before first use of `Launch.bat`.

## Dependency Guardrails (CI/Local)

Use these non-mutating checks to validate CUDA 13 dependency compatibility:

```batch
venv\Scripts\python -m pip install --dry-run --no-deps -r requirements-lock-cu130.txt
venv\Scripts\python -m pip install --dry-run qwen-tts==0.1.1 transformers==4.57.3 huggingface_hub[hf_xet]==0.36.2
venv\Scripts\python verify_install.py --strict-resolver
```
## New Runtime Notes

- Z-Image now supports image-to-image editing (up to 6 reference inputs) in the main UI.
- On ~8GB NVIDIA GPUs (for example RTX 3070), requesting Z-Image Full automatically falls back to Z-Image Quantized for stability.
- Startup now performs dependency profile preflight and selected-model auto-download checks.
- Startup enforces CUDA 13 runtime profile by default on Windows (`UFIG_ENFORCE_CUDA13=1`).
- The app does not apply app-side output censorship; output behavior is model-native.
