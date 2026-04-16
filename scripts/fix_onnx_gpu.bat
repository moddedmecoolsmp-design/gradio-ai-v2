@echo off
echo Uninstalling existing ONNX Runtime...
pip uninstall -y onnxruntime onnxruntime-gpu

echo Installing ONNX Runtime GPU Nightly (CUDA 13 support)...
pip install -U --pre onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/

echo ONNX Runtime GPU fix completed.
pause
