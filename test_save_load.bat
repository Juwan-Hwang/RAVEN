@echo off
cd /d c:\Users\Juwan\Desktop\RAVEN

REM 用 fvecs 数据做 smoke test：--save 然后 --load
REM sift_learn.fvecs 是 100K x 128，太大；用前 1000 个向量

REM Step 1: Build + Save
echo === Step 1: Build + Save ===
target\release\raven_ann_bench --train data\sift\sift_learn.fvecs --save test_index.idx --dim 128 --n 1000 --alpha 1.0 --l-build 20 --r-max 8 --ef-search 50

if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

if not exist test_index.idx (
    echo INDEX FILE NOT CREATED
    exit /b 1
)

echo Index file size:
for %%A in (test_index.idx) do echo %%~zA bytes

REM Step 2: Load + Query
echo.
echo === Step 2: Load + Query ===
target\release\raven_ann_bench --load test_index.idx --train data\sift\sift_learn.fvecs --test data\sift\sift_query.fvecs --output test_output.bin --dim 128 --n 1000 --nq 10 --k 10 --ef-search 50

if errorlevel 1 (
    echo QUERY FAILED
    exit /b 1
)

echo.
echo === SMOKE TEST PASSED ===

del test_index.idx test_output.bin 2>nul
