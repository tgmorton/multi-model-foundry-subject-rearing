@echo off
REM --- Preprocessing Script for RTX 3070 Ti ---
REM This script runs corpus ablations on Windows

REM --- Script Usage ---
if "%1"=="" (
    echo Usage: run_preprocessing.bat ^<config_name_without_yaml_extension^>
    echo Example: run_preprocessing.bat experiment_1_remove_expletives
    exit /b 1
)

set CONFIG_NAME=%1

REM --- Environment Setup ---
echo === Preprocessing Script Started: %date% %time% ===
echo Config: %CONFIG_NAME%
echo Hardware: RTX 3070 Ti
echo.

REM --- Define Paths ---
REM !!! UPDATE THIS TO YOUR PROJECT'S ROOT DIRECTORY !!!
set PROJECT_DIR=C:\Users\Thomas\Documents\subject-drop-rearing
set CONFIG_FILE=configs\%CONFIG_NAME%.yaml

REM --- Preparations ---
echo Project Directory: %PROJECT_DIR%
echo Config File: %CONFIG_FILE%
echo.

REM Check for required files
if not exist "%PROJECT_DIR%\%CONFIG_FILE%" (
    echo ERROR: Config file not found at %PROJECT_DIR%\%CONFIG_FILE%
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

REM Set PyTorch CUDA Allocator Config for better memory management
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

REM --- Run Preprocessing ---
echo ========================================
echo Running: Dataset preprocessing and ablation
echo ========================================
echo.

cd /d "%PROJECT_DIR%"
venv\Scripts\python.exe -m model_foundry.cli preprocess %CONFIG_FILE%

if %ERRORLEVEL% neq 0 (
    echo ERROR: Preprocessing failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Preprocessing Completed: %date% %time%
echo Config: %CONFIG_NAME%
echo Hardware: RTX 3070 Ti
echo ======================================== 