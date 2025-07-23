#!/bin/bash
set -e

echo "Activating virtual environment..."
source myenv/bin/activate

echo " Running demo query..."
python scripts/rag_cli.py --query "What is the mission of Procyon?"
