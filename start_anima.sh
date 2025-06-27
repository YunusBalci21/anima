#!/bin/bash
# ANIMA Quick Start Script

echo "
 █████╗ ███╗   ██╗██╗███╗   ███╗ █████╗
██╔══██╗████╗  ██║██║████╗ ████║██╔══██╗
███████║██╔██╗ ██║██║██╔████╔██║███████║
██╔══██║██║╚██╗██║██║██║╚██╔╝██║██╔══██║
██║  ██║██║ ╚████║██║██║ ╚═╝ ██║██║  ██║
╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝
The Emergent Self Engine
"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import torch" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for .env file
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your Hugging Face token!"
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create necessary directories
mkdir -p experiments static

# Check if static HTML exists
if [ ! -f "static/index.html" ]; then
    echo "📄 Setting up web interface..."
    cp static/index_enhanced.html static/index.html 2>/dev/null || true
fi

# Menu
echo "
🎮 How would you like to run ANIMA?

1) Web Interface (recommended)
2) Headless with LLM
3) Quick Test (no LLM)
4) Consciousness Experiment
5) Creative Explosion
6) Custom Settings

"

read -p "Select option (1-6): " choice

case $choice in
    1)
        echo "🌐 Starting web interface..."
        echo "📱 Open http://localhost:8000 in your browser"
        python anima_main.py --web
        ;;
    2)
        echo "🧠 Starting headless simulation with DeepSeek..."
        python anima_main.py --headless --llm --ticks 1000
        ;;
    3)
        echo "⚡ Running quick test without LLM..."
        python anima_main.py --headless --no-llm --agents 20 --ticks 500
        ;;
    4)
        echo "🔬 Running consciousness evolution experiment..."
        python anima_main.py --experiment consciousness
        ;;
    5)
        echo "🎨 Running creative explosion experiment..."
        python anima_main.py --experiment creativity
        ;;
    6)
        echo "⚙️  Custom settings..."
        read -p "Number of agents (default 20): " agents
        read -p "World size (default 50): " world_size
        read -p "Number of ticks (default 1000): " ticks
        read -p "Use LLM? (y/n, default y): " use_llm

        agents=${agents:-20}
        world_size=${world_size:-50}
        ticks=${ticks:-1000}

        if [[ "$use_llm" == "n" ]]; then
            llm_flag="--no-llm"
        else
            llm_flag="--llm"
        fi

        python anima_main.py --headless $llm_flag --agents $agents --world-size $world_size --ticks $ticks
        ;;
    *)
        echo "Invalid option. Starting web interface..."
        python anima_main.py --web
        ;;
esac