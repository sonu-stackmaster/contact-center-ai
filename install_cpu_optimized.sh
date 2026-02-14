#!/bin/bash

# CPU-Optimized Installation Script for Contact Center AI
# Designed for low-spec systems without GPU support

echo "🚀 Installing Contact Center AI (CPU-Optimized Version)"
echo "System Requirements: Ubuntu 24.04, Python 3.12, 16GB RAM, CPU-only"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ "$python_version" < "3.12" ]]; then
    echo "❌ Python 3.12+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip for better dependency resolution
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install CPU-optimized dependencies in correct order
echo "📚 Installing CPU-optimized dependencies..."

# Install core dependencies first
echo "Installing core FastAPI dependencies..."
pip install fastapi==0.104.1 uvicorn==0.24.0 pydantic==2.5.0

# Install data processing libraries
echo "Installing data processing libraries..."
pip install pandas==2.1.4 "numpy>=1.26.0,<2.0.0"

# Install ML libraries (CPU versions)
echo "Installing ML libraries..."
pip install scikit-learn>=1.4.0

# Install PyTorch CPU version first
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install compatible transformers ecosystem
echo "Installing transformers ecosystem..."
pip install "huggingface-hub>=0.15.0,<1.0.0"
pip install "transformers>=4.21.0,<5.0.0"
pip install "tokenizers>=0.13.0"

# Install sentence transformers with compatible versions
echo "Installing sentence transformers..."
pip install "sentence-transformers>=2.2.0,<3.0.0"

# Install text processing libraries
echo "🧠 Installing lightweight NLP libraries..."
pip install "spacy>=3.7.0,<4.0.0" "nltk>=3.8.0" "textblob>=0.17.0"

# Install vector database (CPU version)
echo "Installing FAISS (CPU version)..."
pip install "faiss-cpu>=1.8.0"

# Install caching and utilities
echo "Installing utilities..."
pip install "diskcache>=5.6.0" python-dotenv==1.0.0 requests==2.31.0

# Install web interface
echo "Installing web interface..."
pip install streamlit==1.29.0 plotly==5.17.0

# Optional: OpenAI for enhanced features
echo "Installing OpenAI client..."
pip install openai==1.6.1

echo "📥 Downloading required language models..."

# Download spaCy model (lightweight)
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data..."
python -c "
import nltk
print('Downloading NLTK data...')
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
print('NLTK data downloaded successfully')
"

# Pre-download sentence transformer model
echo "Pre-downloading sentence transformer model..."
python -c "
try:
    from sentence_transformers import SentenceTransformer
    print('Downloading all-MiniLM-L6-v2 model...')
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    print('Sentence transformer model downloaded successfully')
except Exception as e:
    print(f'Warning: Could not pre-download sentence transformer: {e}')
    print('The application will still work with TF-IDF fallback')
"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data models cache logs

# Set environment variables for CPU optimization
echo "⚙️ Setting CPU optimization flags..."
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# Create optimized environment file
echo "📝 Creating optimized .env file..."
cat > .env.cpu << 'EOF'
# CPU Optimization Settings
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2

# Application Settings
DATABASE_URL=sqlite:///./data/contact_center.db
LOG_LEVEL=INFO

# Optional: Add your OpenAI API key for enhanced features
# OPENAI_API_KEY=your_key_here

# Performance Settings
PYTORCH_ENABLE_MPS_FALLBACK=1
TOKENIZERS_PARALLELISM=false
EOF

# Test the installation
echo "🧪 Testing installation..."
python -c "
import sys
try:
    print('Testing imports...')
    import fastapi
    import numpy
    import pandas
    import sklearn
    import textblob
    import nltk
    import faiss
    print('✅ Core packages imported successfully')
    
    try:
        import sentence_transformers
        print('✅ Sentence transformers available')
    except ImportError:
        print('⚠️ Sentence transformers not available, will use TF-IDF fallback')
    
    print('✅ Installation test passed!')
except Exception as e:
    print(f'❌ Installation test failed: {e}')
    sys.exit(1)
"

echo ""
echo "✅ CPU-optimized installation complete!"
echo ""
echo "🔧 Performance Tips for Low-Spec Systems:"
echo "  • Use smaller batch sizes (already configured)"
echo "  • Enable caching for faster repeated operations"
echo "  • Consider using OpenAI API for heavy LLM tasks"
echo "  • Monitor memory usage with 'htop' or 'free -h'"
echo ""
echo "🚀 To start the application:"
echo "  1. source venv/bin/activate"
echo "  2. cp .env.cpu .env  # Copy optimized settings"
echo "  3. python test_cpu_setup.py  # Test the setup"
echo "  4. python -m app.main  # Start API server"
echo "  5. streamlit run dashboard/streamlit_app.py  # Start dashboard"
echo ""
echo "📊 Expected Performance on Intel i5 11th Gen + 16GB RAM:"
echo "  • API response time: 100-500ms"
echo "  • Intent classification: 50-100ms"
echo "  • Sentiment analysis: 20-50ms"
echo "  • RAG queries: 200-800ms"
echo "  • Memory usage: 2-4GB"