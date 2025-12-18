#!/bin/bash
# Copyright (C) 2024 Delia Contributors
# Setup script for all Delia components (Core, CLI, Dashboard)

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🍈 Starting Delia Full System Setup${NC}"
echo "======================================"

# 1. Check prerequisites
echo -e "\n${BLUE}[1/4] Checking prerequisites...${NC}"

if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv is not installed. Recommended for Python management.${NC}"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm is required but not found. Please install Node.js.${NC}"
    exit 1
fi

# 2. Setup Python Core
echo -e "\n${BLUE}[2/4] Setting up Python Core...${NC}"
if command -v uv &> /dev/null; then
    echo "Using uv for installation..."
    uv sync
    uv pip install -e .
else
    echo "Using pip for installation..."
    pip install -e .
fi
echo -e "${GREEN}✓ Python Core ready${NC}"

# 3. Setup TypeScript CLI
echo -e "\n${BLUE}[3/4] Setting up TypeScript CLI...${NC}"
cd packages/cli
npm install
npm run build
npm link
cd ../..
echo -e "${GREEN}✓ TypeScript CLI built and linked${NC}"

# 4. Setup Dashboard
echo -e "\n${BLUE}[4/4] Setting up Dashboard...${NC}"
cd dashboard
npm install
cd ..
echo -e "${GREEN}✓ Dashboard dependencies installed${NC}"

echo -e "\n${GREEN}======================================"
echo -e "🎉 Delia Setup Complete!"
echo -e "======================================${NC}"
echo -e "\nNext steps:"
echo -e "1. Run ${YELLOW}delia init${NC} to detect your LLMs"
echo -e "2. Run ${YELLOW}delia chat${NC} to start chatting"
echo -e "3. (Optional) cd dashboard && npm run dev to start monitoring"
