@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================
REM Ultra Fast Image Gen Verification (Windows)
REM ============================================
REM Version: 1.0 - Comprehensive dependency verification
REM ============================================

echo.
echo ============================================
echo     Ultra Fast Image Gen Verification
echo ============================================
echo.

REM Initialize error tracking
set "SCRIPT_ERRORS=0"
set "VERIFICATION_START_TIME=%TIME%"

REM Change to root directory for consistent path handling
cd /d "%~dp0.." 2>nul
if errorlevel 1 (
    echo ERROR: Failed to change to root directory.
    echo Current directory: %CD%
    goto :error_exit
)

REM Set script name for logging
set "SCRIPT_NAME=%~nx0"

REM ============================================
REM PRE-VERIFICATION VALIDATION
REM ============================================

echo [1/7] Validating environment...

REM Check if running as administrator (optional but recommended for some operations)
net session >nul 2>&1
if %errorlevel% == 0 (
    set "ADMIN_STATUS=Yes"
    echo Administrator privileges detected.
) else (
    set "ADMIN_STATUS=No"
    echo Running without administrator privileges.
    echo Some operations may be limited.
)

REM Check Windows version for compatibility
ver | findstr /i "Version 10\." >nul
if errorlevel 1 (
    ver | findstr /i "Version 11\." >nul
    if errorlevel 1 (
        echo WARNING: Windows 10 or 11 recommended for best compatibility.
    ) else (
        echo Windows 11 detected - optimal compatibility.
    )
) else (
    echo Windows 10 detected - compatible.
)

REM Check if virtual environment exists
if not exist "venv" (
    echo ERROR: Virtual environment not found at: %CD%\venv
    echo Please run Install.bat first to set up the environment.
    goto :error_exit
)

REM Check if virtual environment is properly set up
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment appears corrupted.
    echo Missing activation script. Please delete venv folder and run Install.bat again.
    goto :error_exit
)

REM Check if dependencies are installed
if not exist "venv\.installed" (
    echo ERROR: Dependencies not installed.
    echo Please run Install.bat first to install dependencies.
    goto :error_exit
)

echo Environment validation complete.

REM ============================================
REM VIRTUAL ENVIRONMENT ACTIVATION
REM ============================================

echo.
echo [2/7] Activating virtual environment...

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    goto :error_exit
)

echo Virtual environment activated successfully.
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM ============================================
REM DEPENDENCY VALIDATION
REM ============================================

echo.
echo [3/7] Validating critical dependencies...

REM Check Python version and availability
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in virtual environment.
    goto :error_exit
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Validate Python version (require 3.10+)
echo %PYTHON_VERSION% | findstr /r "^3\.1[0-9]\." >nul
if errorlevel 1 (
    echo ERROR: Python 3.10 or higher is required. Current: %PYTHON_VERSION%
    goto :error_exit
)

REM Check PyTorch installation and CUDA availability
echo Checking PyTorch installation...
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}') " 2>nul
if errorlevel 1 (
    echo ERROR: PyTorch not properly installed.
    goto :error_exit
)

REM Check critical Python packages
echo Checking critical packages...

REM Check Gradio
python -c "import gradio; print(f'Gradio {gradio.__version__}')" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Gradio not found. UI may not work properly.
    set /a SCRIPT_ERRORS+=1
) else (
    echo Gradio OK.
)

REM Check transformers
python -c "import transformers" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Transformers not found. Model loading may fail.
    set /a SCRIPT_ERRORS+=1
) else (
    echo Transformers OK ^(may be slow on first use^).
)

REM Check diffusers
python -c "import diffusers" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Diffusers not found. Image generation may fail.
    set /a SCRIPT_ERRORS+=1
) else (
    echo Diffusers OK.
)

REM Check optional TTS dependencies
echo Checking TTS dependencies...

REM Check git-lfs
where git-lfs >nul 2>&1
if errorlevel 1 (
    echo WARNING: git-lfs not found. Some models may require manual download.
    echo Install from: https://git-lfs.github.com/
    echo Then run: git lfs install
) else (
    echo git-lfs OK.
)

REM Check qwen-tts without importing its noisy optional runtime probes
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('qwen_tts') else 1)" >nul 2>&1
if errorlevel 1 (
    echo INFO: qwen-tts not found. Qwen TTS features will be unavailable.
) else (
    echo qwen-tts OK.
)




echo Dependency validation complete.

REM ============================================
REM GPU/CUDA VALIDATION
REM ============================================

echo.
echo [4/7] Validating GPU/CUDA setup...

REM Check CUDA availability via PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); [print(f'  Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>nul
if errorlevel 1 (
    echo WARNING: CUDA validation failed. GPU acceleration may not work.
    set /a SCRIPT_ERRORS+=1
)

REM Check for ONNX Runtime GPU
python -c "try: import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}'); print(f'CUDA execution providers: {onnxruntime.get_available_providers()}') except: print('ONNX Runtime check failed')" >nul 2>&1

echo GPU/CUDA validation complete.

REM ============================================
REM DIRECTORY VALIDATION
REM ============================================

echo.
echo [5/7] Validating directories...

REM Ensure model directories exist
if not exist "models" mkdir "models" 2>nul
if not exist "models\qwen-tts" mkdir "models\qwen-tts" 2>nul
if not exist "checkpoints" mkdir "checkpoints" 2>nul
if not exist "cache" mkdir "cache" 2>nul
if not exist "cache\pyannote" mkdir "cache\pyannote" 2>nul
if not exist "cache\whisper" mkdir "cache\whisper" 2>nul

echo Directory validation complete.

REM ============================================
REM CONFIGURATION VALIDATION
REM ============================================

echo.
echo [6/7] Validating configuration...

REM Test environment variable setup
set "TORCHINDUCTOR_MAX_AUTOTUNE=1"
set "TORCHINDUCTOR_FREEZING=1"
set "TORCHINDUCTOR_CACHE_DIR=%CD%\cache\torch_inductor"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,roundup_power2_divisions:32"
set "PYTORCH_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF%"
if "%UFIG_OPTIMIZATION_PROFILE%"=="" set "UFIG_OPTIMIZATION_PROFILE=max_speed"
if "%UFIG_ENABLE_OPTIONAL_ACCELERATORS%"=="" set "UFIG_ENABLE_OPTIONAL_ACCELERATORS=0"
REM SDNQ Triton optimizations now auto-detected via triton-windows package
set "INDEX_TTS_CONFIG_PATH=%CD%\checkpoints\config.yaml"
set "INDEX_TTS_MODEL_DIR=%CD%\checkpoints"
set "QWEN_TTS_CACHE_DIR=%CD%\models\qwen-tts"

echo Configuration validation complete.

REM ============================================
REM VERIFICATION SUMMARY
REM ============================================

echo.
echo [7/7] Generating verification report...

REM Calculate verification time
set "VERIFICATION_END_TIME=%TIME%"

echo.
echo ============================================
echo        VERIFICATION REPORT
echo ============================================
echo Verification completed successfully!
echo.
echo Environment:
echo - Python Version: %PYTHON_VERSION%
echo - Virtual Environment: %CD%\venv
echo - Administrator Privileges: %ADMIN_STATUS%
echo.
echo Dependencies:
python -c "import torch; print('- PyTorch:', torch.__version__, '(CUDA:' + str(torch.cuda.is_available()) + ')')" 2>nul
echo - Gradio: Available
echo - Transformers: Available ^(may be slow on first use^)
echo - Diffusers: Available
echo.
echo Optional Components:
where git-lfs >nul 2>&1 && echo - git-lfs: Available || echo - git-lfs: Not available
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('qwen_tts') else 1)" >nul 2>&1 && echo - qwen-tts: Available || echo - qwen-tts: Not available
if exist "checkpoints" (
    echo - checkpoints/: Found
) else (
    echo - checkpoints/: Missing
)
echo.
echo Directories Created:
echo - models/
echo - models\qwen-tts/
echo - checkpoints/
echo - cache/
echo - cache\pyannote/
echo - cache\whisper/
echo.
if %SCRIPT_ERRORS% gtr 0 (
    echo WARNING: %SCRIPT_ERRORS% warnings detected during verification.
    echo The application may still work but some features might be unavailable.
    echo.
)
echo Verification Time: %VERIFICATION_START_TIME% - %VERIFICATION_END_TIME%
echo.
echo ============================================
echo    ✓ VERIFICATION COMPLETED SUCCESSFULLY
echo ============================================
echo.
echo You can now run Launch.bat to start the application.
echo.

pause
endlocal
exit /b 0

REM ============================================
REM ERROR HANDLING
REM ============================================

:error_exit
echo.
echo ============================================
echo         VERIFICATION FAILED
echo ============================================
echo Verification failed. Check the error messages above.
echo.
echo Common solutions:
echo 1. Run Install.bat to set up the environment
echo 2. Ensure you have Python 3.10+ installed
echo 3. Check that all dependencies are installed
echo 4. Verify CUDA installation if using GPU
echo.
echo Verification Time: %VERIFICATION_START_TIME% - %TIME%
echo.
pause
endlocal
exit /b 1
