@echo off
cd /d c:\Users\Juwan\Desktop\RAVEN
echo === Building release ===
cargo build --release --bin quick_recall_check 2>&1
if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED
    exit /b 1
)
echo === Running baseline ===
target\release\quick_recall_check.exe > baseline_result.txt 2>&1
type baseline_result.txt
