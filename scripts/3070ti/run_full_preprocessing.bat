@echo off
REM --- Full Preprocessing Pipeline Script for RTX 3070 Ti ---
REM This script runs the complete preprocessing pipeline: preprocess → train-tokenizer → tokenize-dataset

REM --- Script Usage ---
if "%1"=="" (
    echo Usage: run_full_preprocessing.bat ^<config_name_without_yaml_extension^>
    echo Example: run_full_preprocessing.bat experiment_1_remove_expletives
    exit /b 1
)

set CONFIG_NAME=%1

REM --- Environment Setup ---
echo === Full Preprocessing Pipeline Started: %date% %time% ===
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

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM === Step 1: Preprocessing ===
echo ========================================
echo Step 1/3: Preprocessing (Corpus Ablations)
echo ========================================
venv\Scripts\python.exe -m model_foundry.cli preprocess %CONFIG_FILE%

if %ERRORLEVEL% neq 0 (
    echo ERROR: Preprocessing failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

REM === Step 2: Train Tokenizer ===
echo ========================================
echo Step 2/3: Training Tokenizer
echo ========================================
venv\Scripts\python.exe -m model_foundry.cli train-tokenizer %CONFIG_FILE%

if %ERRORLEVEL% neq 0 (
    echo ERROR: Tokenizer training failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

REM === Step 3: Tokenize Dataset ===
echo ========================================
echo Step 3/3: Tokenizing Dataset
echo ========================================
venv\Scripts\python.exe -m model_foundry.cli tokenize-dataset %CONFIG_FILE%

if %ERRORLEVEL% neq 0 (
    echo ERROR: Dataset tokenization failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Full Preprocessing Pipeline Completed: %date% %time%
echo Config: %CONFIG_NAME%
echo Hardware: RTX 3070 Ti
echo ========================================
echo.
echo Next steps:
echo 1. Transfer processed data to cluster for training
echo 2. Use cluster scripts for model training
echo 3. Run evaluation on cluster or locally 