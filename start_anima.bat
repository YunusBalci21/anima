@echo off
REM ANIMA Quick Start Script for Windows

echo.
echo  █████╗ ███╗   ██╗██╗███╗   ███╗ █████╗
echo ██╔══██╗████╗  ██║██║████╗ ████║██╔══██╗
echo ███████║██╔██╗ ██║██║██╔████╔██║███████║
echo ██╔══██║██║╚██╗██║██║██║╚██╔╝██║██╔══██║
echo ██║  ██║██║ ╚████║██║██║ ╚═╝ ██║██║  ██║
echo ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝
echo The Emergent Self Engine
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Check if dependencies are installed
python -c "import torch" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Check for .env file
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env file from template...
        copy .env.example .env
        echo Please edit .env and add your Hugging Face token!
    )
)

REM Create necessary directories
if not exist "experiments" mkdir experiments
if not exist "static" mkdir static

REM Check if static HTML exists
if not exist "static\index.html" (
    if exist "static\index_enhanced.html" (
        echo Setting up web interface...
        copy static\index_enhanced.html static\index.html
    )
)

REM Menu
echo.
echo How would you like to run ANIMA?
echo.
echo 1) Web Interface (recommended)
echo 2) Headless with LLM
echo 3) Quick Test (no LLM)
echo 4) Consciousness Experiment
echo 5) Creative Explosion
echo 6) Custom Settings
echo.

set /p choice="Select option (1-6): "

if "%choice%"=="1" (
    echo Starting web interface...
    echo Open http://localhost:8000 in your browser
    python anima_main.py --web
) else if "%choice%"=="2" (
    echo Starting headless simulation with DeepSeek...
    python anima_main.py --headless --llm --ticks 1000
) else if "%choice%"=="3" (
    echo Running quick test without LLM...
    python anima_main.py --headless --no-llm --agents 20 --ticks 500
) else if "%choice%"=="4" (
    echo Running consciousness evolution experiment...
    python anima_main.py --experiment consciousness
) else if "%choice%"=="5" (
    echo Running creative explosion experiment...
    python anima_main.py --experiment creativity
) else if "%choice%"=="6" (
    echo Custom settings...
    set /p agents="Number of agents (default 20): "
    set /p world_size="World size (default 50): "
    set /p ticks="Number of ticks (default 1000): "
    set /p use_llm="Use LLM? (y/n, default y): "

    if "%agents%"=="" set agents=20
    if "%world_size%"=="" set world_size=50
    if "%ticks%"=="" set ticks=1000

    if "%use_llm%"=="n" (
        set llm_flag=--no-llm
    ) else (
        set llm_flag=--llm
    )

    python anima_main.py --headless %llm_flag% --agents %agents% --world-size %world_size% --ticks %ticks%
) else (
    echo Invalid option. Starting web interface...
    python anima_main.py --web
)

pause