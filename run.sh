#!/usr/bin/env bash
# ========================================
#  BASI BOT - RUN SCRIPT (Linux/macOS)
#  Multi-Agent Discord LLM Chatbot System
# ========================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo " BASI BOT - LAUNCH SCRIPT"
echo " Multi-Agent Discord LLM Chatbot System"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ========================================
# Check Virtual Environment
# ========================================
if [ ! -d "venv" ]; then
    echo -e "${RED}ERROR: Virtual environment not found!${NC}"
    echo "Please run setup.sh first to set up the environment."
    echo ""
    echo "  ./setup.sh"
    echo ""
    exit 1
fi

# ========================================
# Activate Virtual Environment
# ========================================
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to activate virtual environment${NC}"
    exit 1
fi
echo ""

# ========================================
# Check FFmpeg (warning only)
# ========================================
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}WARNING: FFmpeg not found. Video features will be disabled.${NC}"
    echo "Install FFmpeg to enable video generation."
    echo ""
fi

# ========================================
# Start Application
# ========================================
echo -e "${GREEN}Starting BASI Bot...${NC}"
echo "Gradio UI will open in your browser automatically."
echo ""

python main.py

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}ERROR: Application exited with code $EXIT_CODE${NC}"
    echo "Check the error messages above."
    echo ""
fi
