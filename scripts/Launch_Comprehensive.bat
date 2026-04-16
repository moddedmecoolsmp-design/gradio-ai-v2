@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================
REM Ultra Fast Image Gen Launcher (Windows)
REM ============================================
REM Version: 2.0 - Comprehensive fixes and improvements
REM ============================================

echo.
echo ============================================
echo        Ultra Fast Image Gen (Windows)
echo ============================================
echo.

REM Initialize error tracking
set "SCRIPT_ERRORS=0"
set "LAUNCH_START_TIME=%TIME%"

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
REM PRE-LAUNCH VALIDATION
REM ============================================

echo [1/8] Validating environment...

REM Check if running as administrator (optional but recommended for some operations)
net session >nul 2>&1
if %errorlevel% == 0 (
    echo Administrator privileges detected.
) else (
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
echo [2/8] Activating virtual environment...

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    goto :error_exit
)

echo Virtual environment activated successfully.

REM ============================================
REM DEPENDENCY VALIDATION
REM ============================================

echo.
echo [3/8] Validating critical dependencies...

REM Check Python version and availability
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in virtual environment.
    goto :error_exit
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Validate Python version (require 3.10+)
echo %PYTHON_VERSION% | findstr /r "^3\.[1-9][0-9]*\." >nul
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
echo [DEBUG] Starting package validation...

REM Check Gradio
python -c "import gradio; print(f'Gradio {gradio.__version__}')" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Gradio not found. UI may not work properly.
    set /a SCRIPT_ERRORS+=1
) else (
    echo Gradio OK.
)

REM Check transformers (simplified to avoid hanging)
echo [DEBUG] Checking transformers...
echo {"location":"Launch.bat:142","message":"Checking transformers import","data":{},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
python -c "import transformers" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Transformers not found. Model loading may fail.
    set /a SCRIPT_ERRORS+=1
    echo {"location":"Launch.bat:147","message":"Transformers check failed","data":{"errorlevel":"%errorlevel%"},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
) else (
    echo Transformers OK ^(may be slow on first use^).
    echo {"location":"Launch.bat:149","message":"Transformers check passed","data":{},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
)

REM Check diffusers
echo [DEBUG] Checking diffusers...
echo {"location":"Launch.bat:153","message":"Checking diffusers import","data":{},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
python -c "import diffusers" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Diffusers not found. Image generation may fail.
    set /a SCRIPT_ERRORS+=1
    echo {"location":"Launch.bat:158","message":"Diffusers check failed","data":{"errorlevel":"%errorlevel%"},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
) else (
    echo Diffusers OK.
    echo {"location":"Launch.bat:160","message":"Diffusers check passed","data":{},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
)

REM Check optional TTS dependencies
echo Checking TTS dependencies...
echo {"location":"Launch.bat:164","message":"Starting TTS dependency checks","data":{},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
echo [DEBUG] Starting TTS checks...

echo [DEBUG] Checking git-lfs...
echo {"location":"Launch.bat:git-lfs","message":"Checking git-lfs","data":{},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
where git-lfs >nul 2>&1
if errorlevel 1 (
    echo Install from: https://git-lfs.github.com/
    echo Then run: git lfs install
    echo {"location":"Launch.bat:git-lfs","message":"git-lfs not found","data":{"errorlevel":"%errorlevel%"},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
) else (
    echo git-lfs OK.
    echo {"location":"Launch.bat:git-lfs","message":"git-lfs check passed","data":{},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
)

REM Check qwen-tts
echo [DEBUG] Checking qwen-tts...
echo {"location":"Launch.bat:qwen-tts","message":"Checking qwen-tts import","data":{},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
python -c "import qwen_tts" >nul 2>&1
if errorlevel 1 (
    echo INFO: qwen-tts not found. Qwen TTS features will be unavailable.
    echo {"location":"Launch.bat:qwen-tts","message":"qwen-tts not found","data":{"errorlevel":"%errorlevel%"},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
) else (
    echo qwen-tts OK.
    echo {"location":"Launch.bat:qwen-tts","message":"qwen-tts check passed","data":{},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"
)

if exist "checkpoints" (
    python -c "import index_tts" >nul 2>&1
    if errorlevel 1 (
    ) else (
    )
) else (
)

echo Dependency validation complete.

REM ============================================
REM PORT AND PROCESS CLEANUP
REM ============================================

echo.
echo [4/8] Cleaning up existing processes...
echo {"location":"Launch.bat:port-cleanup","message":"Starting port cleanup","data":{},"timestamp":"%time%","sessionId":"debug-session","runId":"test-launch"} >> "%~dp0.cursor\debug.log"

set "GRADIO_PORT=7860"
set "SERVER_PID="

REM Check if port is in use and get process information
echo Checking if port %GRADIO_PORT% is in use...
powershell -Command "try { $conn = Get-NetTCPConnection -LocalPort %GRADIO_PORT% -State Listen -ErrorAction SilentlyContinue; if ($conn) { Write-Host 'Port %GRADIO_PORT% is in use by process:' $conn.OwningProcess; Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2; Write-Host 'Process terminated.' } else { Write-Host 'No server running on port %GRADIO_PORT%' } } catch { Write-Host 'Port check completed' }" 2>nul
if errorlevel 1 (
    echo WARNING: Port cleanup may have failed
)

REM Additional cleanup: Kill any orphaned python processes that might be running app.py
taskkill /F /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq app.py*" /FI "MEMUSAGE gt 50" >nul 2>&1
taskkill /F /FI "IMAGENAME eq python.exe" /FI "CPUTIME gt 00:00:30" >nul 2>&1

REM Wait a moment for processes to fully terminate
timeout /t 2 /nobreak >nul

echo Process cleanup complete.

REM ============================================
REM ENVIRONMENT CONFIGURATION
REM ============================================

echo.
echo [5/8] Configuring environment variables...

REM PyTorch optimizations
set "TORCHINDUCTOR_MAX_AUTOTUNE=1"
set "TORCHINDUCTOR_FREEZING=1"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
set "PYTORCH_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF%"
if "%UFIG_OPTIMIZATION_PROFILE%"=="" set "UFIG_OPTIMIZATION_PROFILE=max_speed"
if "%UFIG_ENABLE_OPTIONAL_ACCELERATORS%"=="" set "UFIG_ENABLE_OPTIONAL_ACCELERATORS=1"

REM HuggingFace optimizations
set "HF_HUB_DISABLE_PROGRESS_BARS=0"
set "HF_HUB_ENABLE_HF_TRANSFER=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM Model-specific configurations

set "INDEX_TTS_CONFIG_PATH=%CD%\checkpoints\config.yaml"
set "INDEX_TTS_MODEL_DIR=%CD%\checkpoints"
set "INDEX_TTS_USE_FP16=true"
set "INDEX_TTS_USE_CUDA_KERNEL=true"
set "INDEX_TTS_AUTO_INSTALL=true"
set "INDEX_TTS_UNLOAD_AFTER_GEN=false"

REM qwen-tts configuration
set "QWEN_TTS_CACHE_DIR=%CD%\models\qwen-tts"
set "QWEN_TTS_DTYPE=auto"
set "QWEN_TTS_UNLOAD_AFTER_GEN=false"

REM Audio processing configuration
set "PYANNOTE_CACHE_DIR=%CD%\cache\pyannote"
set "WHISPER_CACHE_DIR=%CD%\cache\whisper"

REM Ensure model directories exist
if not exist "models" mkdir "models" 2>nul
if not exist "models\qwen-tts" mkdir "models\qwen-tts" 2>nul
if not exist "checkpoints" mkdir "checkpoints" 2>nul
if not exist "cache" mkdir "cache" 2>nul
if not exist "cache\pyannote" mkdir "cache\pyannote" 2>nul
if not exist "cache\whisper" mkdir "cache\whisper" 2>nul

echo Environment configuration complete.

REM ============================================
REM GPU/CUDA VALIDATION
REM ============================================

echo.
echo [6/8] Validating GPU/CUDA setup...

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
REM LAUNCH APPLICATION
REM ============================================

echo.
echo [7/8] Starting Ultra Fast Image Gen...

REM Calculate launch time
set "LAUNCH_TIME=%TIME%"

REM Display final configuration summary
echo.
echo ============================================
echo           LAUNCH CONFIGURATION
echo ============================================
echo Python Version: %PYTHON_VERSION%
echo Virtual Environment: %CD%\venv
echo Gradio Port: %GRADIO_PORT%
echo CUDA Available: Checking...
python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>nul
echo Local URL: http://localhost:%GRADIO_PORT%
echo ============================================
echo.

REM Warn about potential issues
if %SCRIPT_ERRORS% gtr 0 (
    echo WARNING: %SCRIPT_ERRORS% configuration warnings detected.
    echo The application may still work but some features might be unavailable.
    echo.
)

echo Starting Gradio UI...
echo Press Ctrl+C to stop the server
echo.

REM Open browser after a short delay (in background)
start /b timeout /t 5 /nobreak >nul && start "" "http://localhost:%GRADIO_PORT%"

REM Launch the application
echo [%DATE% %TIME%] Starting app.py...
python app.py
set "APP_EXIT_CODE=%errorlevel%"

REM ============================================
REM POST-LAUNCH CLEANUP
REM ============================================

echo.
echo [8/8] Application exited with code: %APP_EXIT_CODE%

REM Calculate runtime
set "LAUNCH_END_TIME=%TIME%"

REM Perform cleanup
echo Performing post-launch cleanup...

REM Kill any remaining processes on the port
powershell -Command "$ErrorActionPreference = 'SilentlyContinue'; try { $conn = Get-NetTCPConnection -LocalPort %GRADIO_PORT% -State Listen; if ($conn) { Stop-Process -Id $conn.OwningProcess -Force } } catch { }" 2>nul

REM Clean up temporary files if any
if exist "temp" rmdir /s /q "temp" 2>nul

REM Exit handling
if %APP_EXIT_CODE% neq 0 (
    echo.
    echo ============================================
    echo           APPLICATION ERROR
    echo ============================================
    echo The application exited with an error code: %APP_EXIT_CODE%
    echo.
    echo Possible causes:
    echo - Port %GRADIO_PORT% is already in use by another application
    echo - Insufficient system resources (RAM/GPU memory)
    echo - Missing or corrupted model files
    echo - Incompatible CUDA/driver versions
    echo - Network connectivity issues (for model downloads)
    echo.
    echo Troubleshooting steps:
    echo 1. Restart your computer
    echo 2. Run Install.bat again to verify dependencies
    echo 3. Check Windows Event Viewer for system errors
    echo 4. Ensure you have sufficient disk space (^>50GB free)
    echo 5. Update your GPU drivers if using CUDA
    echo.
    goto :error_exit
) else (
    echo.
    echo Application exited normally.
    goto :success_exit
)

REM ============================================
REM ERROR HANDLING
REM ============================================

:error_exit
echo.
echo ============================================
echo              LAUNCH FAILED
echo ============================================
echo Script execution failed. Check the error messages above.
echo.
echo Runtime: %LAUNCH_START_TIME% - %TIME%
echo.
pause
endlocal
exit /b 1

REM ============================================
REM SUCCESS HANDLING
REM ============================================

:success_exit
echo.
echo ============================================
echo             LAUNCH SUCCESSFUL
echo ============================================
echo Application completed successfully.
echo Runtime: %LAUNCH_START_TIME% - %TIME%
echo.
endlocal
exit /b 0
