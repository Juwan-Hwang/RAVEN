@echo off
chcp 65001 >nul 2>&1
cd /d c:\Users\Juwan\Desktop\RAVEN
git config user.email "Juwan-Hwang@users.noreply.github.com"
git config user.name "Juwan Hwang"
git add src\config\config.rs src\config\rules.rs src\graph\vamana.rs
git add "RAVEN世界第一冲刺计划.md"
git commit -m "fix: S6 config avq+L2 conflict + init_random_graph deadlock - 155 tests passed"
