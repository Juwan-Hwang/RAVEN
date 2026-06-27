@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d c:\Users\Juwan\Desktop\RAVEN
cargo %*
