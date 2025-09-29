#!/bin/bash

# LegalKit Web Interface Startup Script

echo "===== LegalKit Web Interface Setup ====="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run this script from the web directory."
    exit 1
fi

# Create output directory
mkdir -p web_run_output

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Python version: $python_version"

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Not running in a virtual environment"
    echo "Consider using: python -m venv venv && source venv/bin/activate"
else
    echo "Virtual environment: $VIRTUAL_ENV"
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install requirements"
        exit 1
    fi
else
    echo "Warning: requirements.txt not found"
fi

# Check if LegalKit is installed
echo "Checking LegalKit installation..."
python3 -c "import sys; sys.path.insert(0, '..'); import legalkit; print('LegalKit successfully imported')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: LegalKit import failed. Make sure you're in the correct directory."
    echo "Current directory: $(pwd)"
    echo "Expected to be in: /path/to/LegalKit/web"
fi

# Check if torch is available
echo "Checking PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: PyTorch not found or not properly installed"
fi

# Check port availability
echo "Checking if port 5000 is available..."
if command -v netstat >/dev/null 2>&1; then
    if netstat -tuln | grep -q ":5000 "; then
        echo "Warning: Port 5000 appears to be in use"
        echo "You may need to stop other services or change the port"
    fi
fi

echo ""
echo "===== Starting LegalKit Web Interface ====="
echo "Server will be available at: http://localhost:5000"
echo "Alternative access: http://127.0.0.1:5000"
echo ""
echo "Features:"
echo "  - Interactive model evaluation"
echo "  - Real-time task monitoring"  
echo "  - Results visualization"
echo "  - Multi-dataset support"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
python3 app.py