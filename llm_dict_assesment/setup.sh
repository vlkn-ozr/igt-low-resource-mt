#!/bin/bash

echo "🚀 Dictionary Assessment Tool Setup"
echo "=================================="

python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
min_version="3.8"

if [ "$(printf '%s\n' "$min_version" "$python_version" | sort -V | head -n1)" = "$min_version" ]; then 
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python $python_version is too old. Please use Python 3.8+"
    exit 1
fi

echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

if [ -f "dict_llm_5k_lemma_lowercase.txt" ]; then
    echo "✅ Dictionary file found"
else
    echo "⚠️  Dictionary file not found. Please ensure dict_llm_5k_lemma_lowercase.txt is in the current directory"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "🔥 GPU detected, will use CUDA acceleration"
else
    echo "💻 No GPU detected, will use CPU (slower but works)"
fi

echo ""
echo "🎯 Setup complete! You can now run:"
echo "   python dict_assessment_script.py --test-run"
echo ""
echo "💡 For full processing:"
echo "   python dict_assessment_script.py"
echo ""

read -p "Would you like to run a test with 5 pairs now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 Running test with 5 pairs..."
    python dict_assessment_script.py --max-pairs 5
fi 