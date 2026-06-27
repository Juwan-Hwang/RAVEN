@echo off
tasklist | findstr /i "quick_recall cargo raven rustc"
echo EXIT_CODE=%errorlevel%
