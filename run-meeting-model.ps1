Set-Location -LiteralPath $PSScriptRoot
& .\.venv-wlasl\python.exe -u .\backend\sign_inference\camera_inference.py --model meeting100 --landmarks @args
