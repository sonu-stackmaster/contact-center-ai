#!/usr/bin/env python3
"""
Test script to verify CPU-optimized setup works correctly
"""

import sys
import time
import traceback
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import sklearn
        print("✅ Scikit-learn imported successfully")
        
        import sentence_transformers
        print("✅ Sentence Transformers imported successfully")
        
        import textblob
        print("✅ TextBlob imported successfully")
        
        import nltk
        print("✅ NLTK imported successfully")
        
        import faiss
        print("✅ FAISS imported successfully")
        
        import fastapi
        print("✅ FastAPI imported successfully")
        
        import streamlit
        print("✅ Streamlit imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_models():
    """Test that lightweight models can be loaded."""
    print("\n🧠 Testing model loading...")
    
    try:
        # Test sentence transformer
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("✅ Sentence Transformer model loaded")
        
        # Test embedding generation
        embeddings = model.encode(["This is a test sentence"], show_progress_bar=False)
        print(f"✅ Embedding generated: shape {embeddings.shape}")
        
        # Test TextBlob
        from textblob import TextBlob
        blob = TextBlob("This is a positive sentence")
        sentiment = blob.sentiment.polarity
        print(f"✅ TextBlob sentiment analysis: {sentiment}")
        
        # Test NLTK
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            print("✅ NLTK punkt tokenizer available")
        except LookupError:
            print("⚠️ NLTK punkt tokenizer not found, downloading...")
            nltk.download('punkt', quiet=True)
            print("✅ NLTK punkt tokenizer downloaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        traceback.print_exc()
        return False

def test_services():
    """Test that core services can be initialized."""
    print("\n⚙️ Testing service initialization...")
    
    try:
        # Add app directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Test ML Service
        from app.services.ml_service import MLService
        ml_service = MLService()
        print("✅ ML Service initialized")
        
        # Test sentiment prediction
        sentiment, confidence = ml_service.predict_sentiment("This is a great product!")
        print(f"✅ Sentiment prediction: {sentiment} (confidence: {confidence:.2f})")
        
        # Test RAG Service
        from app.services.rag_service import RAGService
        rag_service = RAGService()
        print("✅ RAG Service initialized")
        
        # Test RAG query
        result = rag_service.query("What is your refund policy?")
        print(f"✅ RAG query successful: {len(result['answer'])} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Service initialization error: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """Test performance on sample tasks."""
    print("\n⚡ Testing performance...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        
        from app.services.ml_service import MLService
        from app.services.rag_service import RAGService
        
        ml_service = MLService()
        rag_service = RAGService()
        
        # Test sentiment analysis speed
        start_time = time.time()
        for i in range(10):
            ml_service.predict_sentiment(f"This is test message number {i}")
        sentiment_time = (time.time() - start_time) * 1000 / 10
        print(f"✅ Average sentiment analysis time: {sentiment_time:.1f}ms")
        
        # Test RAG query speed
        start_time = time.time()
        rag_service.query("How do I return an item?")
        rag_time = (time.time() - start_time) * 1000
        print(f"✅ RAG query time: {rag_time:.1f}ms")
        
        # Performance assessment
        if sentiment_time < 100 and rag_time < 1000:
            print("🚀 Performance: Excellent for low-spec system")
        elif sentiment_time < 200 and rag_time < 2000:
            print("👍 Performance: Good for low-spec system")
        else:
            print("⚠️ Performance: May need optimization")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage."""
    print("\n💾 Testing memory usage...")
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        print(f"📊 Current memory usage: {memory_mb:.1f} MB")
        
        if memory_mb < 500:
            print("✅ Memory usage: Excellent")
        elif memory_mb < 1000:
            print("👍 Memory usage: Good")
        elif memory_mb < 2000:
            print("⚠️ Memory usage: Moderate")
        else:
            print("❌ Memory usage: High - consider optimization")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 CPU-Optimized Contact Center AI - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Model Loading", test_models),
        ("Service Initialization", test_services),
        ("Performance", test_performance),
        ("Memory Usage", test_memory_usage)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your CPU-optimized setup is ready.")
        print("\n🚀 Next steps:")
        print("  1. Start the API: python -m app.main")
        print("  2. Launch dashboard: streamlit run dashboard/streamlit_app.py")
        print("  3. Monitor performance: python scripts/monitor_performance.py --single")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("💡 Try running the installation script: ./install_cpu_optimized.sh")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)