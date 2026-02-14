# Contact Center AI - CPU Optimized Edition

An AI-powered customer support analytics and assistance platform optimized for low-spec, CPU-only systems.

## 🎯 System Requirements

### Minimum Requirements (CPU-Only)
- **OS**: Ubuntu 24.04 (or similar Linux distribution)
- **Python**: 3.12+
- **CPU**: Intel i5 11th Gen or equivalent
- **RAM**: 16GB (8GB minimum)
- **Storage**: 10GB free space
- **GPU**: Not required (CPU-only optimization)

### Recommended for Better Performance
- **RAM**: 32GB
- **CPU**: Intel i7 or AMD Ryzen 7
- **SSD**: For faster model loading

## 🚀 Quick Start (CPU-Optimized)

### Option 1: Automated Installation
```bash
# Clone the repository
git clone <repository-url>
cd contact-center-ai

# Run CPU-optimized installation
./install_cpu_optimized.sh

# Activate environment and start
source venv/bin/activate
cp .env.cpu .env
python -m app.main
```

### Option 2: Manual Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CPU-optimized dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Download language models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"

# Set up environment
cp .env.example .env
mkdir -p data models cache logs

# Start the application
python -m app.main
```

### Option 3: Docker (CPU-Optimized)
```bash
# Build CPU-optimized image
docker build -t contact-center-ai-cpu .

# Run with CPU optimization
docker run -p 8000:8000 -p 8501:8501 \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  --memory=8g \
  --cpus=2 \
  contact-center-ai-cpu
```

## 🔧 CPU Optimization Features

### Lightweight Models
- **Sentence Transformers**: `all-MiniLM-L6-v2` (22MB)
- **Sentiment Analysis**: TextBlob + NLTK (CPU-only)
- **Intent Classification**: Scikit-learn TF-IDF + Logistic Regression
- **Text Processing**: spaCy `en_core_web_sm` (15MB)

### Performance Optimizations
- **Caching**: Disk-based caching for embeddings and predictions
- **Batch Processing**: Optimized batch sizes for limited RAM
- **Thread Limiting**: Controlled thread usage for CPU efficiency
- **Memory Management**: Reduced model sizes and smart garbage collection

### Expected Performance (Intel i5 11th Gen + 16GB RAM)
- **API Response Time**: 100-500ms
- **Intent Classification**: 50-100ms
- **Sentiment Analysis**: 20-50ms
- **RAG Queries**: 200-800ms
- **Memory Usage**: 2-4GB
- **CPU Usage**: 30-60% under normal load

## 📊 Performance Monitoring

Monitor system performance and get optimization recommendations:

```bash
# Single performance check
python scripts/monitor_performance.py --single

# Continuous monitoring (5 minutes)
python scripts/monitor_performance.py --duration 5

# Extended monitoring with custom interval
python scripts/monitor_performance.py --duration 30 --interval 60
```

## 🏗️ Architecture

### Core Components
- **FastAPI Backend**: Lightweight REST API
- **Streamlit Dashboard**: Web-based analytics interface
- **SQLite Database**: Local data storage
- **FAISS Vector Store**: CPU-optimized similarity search

### Services
- **ML Service**: Intent classification and sentiment analysis
- **RAG Service**: Knowledge base querying with context retrieval
- **LLM Service**: OpenAI integration with local fallbacks
- **Agent Assist**: Real-time conversation analysis

## 🔌 API Endpoints

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Analyze Conversation
```bash
curl -X POST http://localhost:8000/api/v1/conversations/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "customer_message": "I need help with my order",
    "agent_response": "I will help you with that right away"
  }'
```

### RAG Query
```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is your refund policy?"}'
```

### Generate Response
```bash
curl -X POST http://localhost:8000/api/v1/agent/suggest-response \
  -H "Content-Type: application/json" \
  -d '{
    "customer_message": "I am frustrated with my order delay",
    "context": "Customer ordered 3 days ago"
  }'
```

## 🎛️ Configuration

### Environment Variables (.env)
```bash
# Database
DATABASE_URL=sqlite:///./data/contact_center.db

# Logging
LOG_LEVEL=INFO

# Optional: OpenAI API (for enhanced features)
OPENAI_API_KEY=your_key_here

# CPU Optimization
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
TOKENIZERS_PARALLELISM=false
```

### Performance Tuning
Edit `app/utils/config.py` for fine-tuning:
```python
# Reduce for lower memory usage
CHUNK_SIZE = 300
BATCH_SIZE = 16
CACHE_SIZE = 100  # MB

# Increase for better accuracy (uses more resources)
MAX_SEQUENCE_LENGTH = 256
TOP_K_RESULTS = 3
```

## 📈 Usage Examples

### 1. Start the API Server
```bash
source venv/bin/activate
python -m app.main
```

### 2. Launch Dashboard
```bash
# In a new terminal
source venv/bin/activate
streamlit run dashboard/streamlit_app.py
```

### 3. Generate Sample Data
```bash
python scripts/generate_data.py --count 100
```

### 4. Monitor Performance
```bash
python scripts/monitor_performance.py --single
```

## 🔍 Troubleshooting

### Common Issues

**High Memory Usage**
```bash
# Reduce cache size in config.py
CACHE_SIZE = 50  # Reduce from 100MB

# Clear cache
rm -rf cache/*

# Restart application
```

**Slow Response Times**
```bash
# Enable OpenAI API for heavy tasks
export OPENAI_API_KEY=your_key

# Reduce batch sizes
BATCH_SIZE = 8  # Reduce from 16

# Use fewer threads
export OMP_NUM_THREADS=1
```

**Model Loading Errors**
```bash
# Re-download models
python -c "
import nltk
from sentence_transformers import SentenceTransformer
nltk.download('punkt', force=True)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
"
```

### Performance Tips

1. **Use OpenAI API**: For complex LLM tasks, use OpenAI API instead of local models
2. **Enable Caching**: Keep caching enabled for repeated operations
3. **Monitor Resources**: Use the performance monitor to identify bottlenecks
4. **Optimize Batch Sizes**: Reduce batch sizes if running out of memory
5. **Close Other Apps**: Free up RAM by closing unnecessary applications

## 🛠️ Development

### Project Structure
```
contact-center-ai/
├── app/
│   ├── api/           # FastAPI endpoints
│   ├── data/          # Database management
│   ├── models/        # Data schemas
│   ├── rag/           # Knowledge base & retrieval
│   ├── services/      # Core business logic
│   └── utils/         # Configuration & utilities
├── dashboard/         # Streamlit web interface
├── scripts/           # Utility scripts
├── data/             # Database files
├── models/           # Trained ML models
└── cache/            # Performance cache
```

### Adding New Features
1. **New API Endpoint**: Add to `app/api/endpoints.py`
2. **New Service**: Create in `app/services/`
3. **New Model**: Add to `app/models/schemas.py`
4. **Dashboard Component**: Extend `dashboard/streamlit_app.py`

### Testing
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/

# Test specific endpoint
python scripts/monitor_performance.py --single
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on low-spec hardware
5. Submit a pull request

## 📞 Support

For issues related to CPU optimization or low-spec deployment:
- Check the troubleshooting section
- Run performance monitoring
- Review system requirements
- Consider using OpenAI API for heavy tasks

---

**Note**: This CPU-optimized version is specifically designed for systems without GPU support. For GPU-enabled systems, consider the standard version with larger transformer models.