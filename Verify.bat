@echo off
setlocal EnableExtensions

REM ============================================
REM Ultra Fast Image Gen Verification (Windows)
REM ============================================
REM Version: 1.0 - Quick verification launcher
REM ============================================

echo.
echo ============================================
echo     Ultra Fast Image Gen Verification
echo ============================================
echo.
echo Running comprehensive verification...
echo.

REM Change to script directory and run verification
cd /d "%~dp0" 2>nul
if errorlevel 1 (
    echo ERROR: Failed to change to script directory.
    goto :error_exit
)

REM Run the comprehensive verification script
call scripts\Verify.bat
set VERIFY_EXIT_CODE=%errorlevel%

if %VERIFY_EXIT_CODE% neq 0 (
    echo.
    echo ============================================
    echo         VERIFICATION FAILED
    echo ============================================
    echo.
    echo Please fix the issues above before running Launch.bat
    echo.
    pause
    endlocal
    exit /b 1
)

echo.
echo ============================================
echo         VERIFICATION PASSED
echo ============================================
echo.
echo You can now run Launch.bat to start the application.
echo.
pause
endlocal
exit /b 0

:error_exit
echo.
echo ============================================
echo         VERIFICATION ERROR
echo ============================================
echo.
pause
endlocal
exit /b 1
