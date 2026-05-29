#!/bin/zsh
# rolleRcoasteR quick launcher

set -e

echo "🎢 Starting rolleRcoasteR..."

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Ensuring dependencies are installed..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "Launching the rollercoaster..."
streamlit run app.py --server.headless true
