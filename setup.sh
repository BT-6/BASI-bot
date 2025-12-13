#!/usr/bin/env bash
# ========================================
#  BASI BOT - SETUP SCRIPT (Linux/macOS)
#  Multi-Agent Discord LLM Chatbot System
# ========================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo " BASI BOT - SETUP SCRIPT"
echo " Multi-Agent Discord LLM Chatbot System"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)     PLATFORM="Linux";;
    Darwin*)    PLATFORM="macOS";;
    *)          PLATFORM="Unknown";;
esac

echo -e "${BLUE}Detected platform: ${PLATFORM}${NC}"
echo ""

# ========================================
# Step 1: Check Python
# ========================================
echo -e "${BLUE}[1/6] Checking Python installation...${NC}"

# Try python3 first, then python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}ERROR: Python not found!${NC}"
    echo "Please install Python 3.8 or higher."
    if [ "$PLATFORM" = "Linux" ]; then
        echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
        echo "  Fedora: sudo dnf install python3 python3-pip"
        echo "  Arch: sudo pacman -S python python-pip"
    elif [ "$PLATFORM" = "macOS" ]; then
        echo "  Using Homebrew: brew install python3"
        echo "  Or download from: https://www.python.org/downloads/"
    fi
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}ERROR: Python 3.8 or higher required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}Found Python $PYTHON_VERSION ($PYTHON_CMD)${NC}"
echo ""

# ========================================
# Step 2: Create Virtual Environment
# ========================================
echo -e "${BLUE}[2/6] Creating Python virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
else
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to create virtual environment${NC}"
        echo "You may need to install python3-venv:"
        echo "  Ubuntu/Debian: sudo apt install python3-venv"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created successfully.${NC}"
fi
echo ""

# ========================================
# Step 3: Activate and Upgrade pip
# ========================================
echo -e "${BLUE}[3/6] Activating environment and upgrading pip...${NC}"

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet
echo -e "${GREEN}pip upgraded successfully.${NC}"
echo ""

# ========================================
# Step 4: Install Requirements
# ========================================
echo -e "${BLUE}[4/6] Installing required packages...${NC}"

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ERROR: requirements.txt not found!${NC}"
    exit 1
fi

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install packages${NC}"
    exit 1
fi
echo -e "${GREEN}All packages installed successfully.${NC}"
echo ""

# ========================================
# Step 5: Check/Install FFmpeg
# ========================================
echo -e "${BLUE}[5/6] Checking FFmpeg installation...${NC}"

FFMPEG_INSTALLED=false
FFPROBE_INSTALLED=false

# Check if FFmpeg is available
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1)
    echo -e "${GREEN}FFmpeg found: $FFMPEG_VERSION${NC}"
    FFMPEG_INSTALLED=true
fi

if command -v ffprobe &> /dev/null; then
    echo -e "${GREEN}FFprobe found${NC}"
    FFPROBE_INSTALLED=true
fi

if [ "$FFMPEG_INSTALLED" = false ]; then
    echo -e "${YELLOW}FFmpeg not found. Video features require FFmpeg.${NC}"
    echo ""
    echo "To install FFmpeg:"
    if [ "$PLATFORM" = "Linux" ]; then
        echo "  Ubuntu/Debian: sudo apt install ffmpeg"
        echo "  Fedora: sudo dnf install ffmpeg"
        echo "  Arch: sudo pacman -S ffmpeg"
    elif [ "$PLATFORM" = "macOS" ]; then
        echo "  Using Homebrew: brew install ffmpeg"
    fi
    echo ""
    echo -e "${YELLOW}You can continue without FFmpeg, but video generation will be disabled.${NC}"
fi

# Create bin directory for consistency (even if using system FFmpeg)
mkdir -p bin
echo ""

# ========================================
# Step 6: Create Required Directories
# ========================================
echo -e "${BLUE}[6/6] Creating required directories...${NC}"

mkdir -p config
mkdir -p data
mkdir -p data/video_temp

echo -e "${GREEN}Directories created.${NC}"
echo ""

# ========================================
# Summary
# ========================================
echo "========================================"
echo -e "${GREEN} SETUP COMPLETE${NC}"
echo "========================================"
echo ""
echo "FFmpeg Status:"
if [ "$FFMPEG_INSTALLED" = true ]; then
    echo -e "  ${GREEN}[OK]${NC} ffmpeg installed (system)"
else
    echo -e "  ${YELLOW}[!!]${NC} ffmpeg NOT FOUND - video features disabled"
fi
if [ "$FFPROBE_INSTALLED" = true ]; then
    echo -e "  ${GREEN}[OK]${NC} ffprobe installed (system)"
else
    echo -e "  ${YELLOW}[!!]${NC} ffprobe NOT FOUND"
fi
echo ""
echo "To start the bot, run:"
echo -e "  ${BLUE}./run.sh${NC}"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
