#!/bin/bash

# Reception Robot - Quick Start Script

echo "🤖 Reception Robot System"
echo "=========================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "✓ Python version: $python_version"

# Check if tkinter is available
if python3 -c "import tkinter" 2>/dev/null; then
    echo "✓ Tkinter available"
else
    echo "✗ Tkinter NOT available"
    echo ""
    echo "Please install tkinter:"
    echo "  Ubuntu/Debian: sudo apt-get install python3-tk"
    echo "  macOS: brew install python-tk"
    echo "  Windows: Reinstall Python with tkinter option"
    exit 1
fi

echo ""
echo "Starting Reception Robot System..."
echo "=================================="
echo ""

# Run the main application
python3 main.py

echo ""
echo "System stopped."
