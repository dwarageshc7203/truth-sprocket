# AI Content Detection Backend - Deployment Guide

## Overview
This Python ML backend provides real AI content detection for Instagram and YouTube short-form content using state-of-the-art transformer models.

## Features
- **Real-time AI Detection**: Uses fine-tuned RoBERTa models for accurate detection
- **Social Media Integration**: Extracts content from Instagram posts and YouTube videos
- **Ensemble Prediction**: Combines multiple models and heuristics for better accuracy
- **Batch Processing**: Support for analyzing multiple URLs simultaneously
- **Custom Training**: Train your own models with collected data

## Installation

### Local Development

1. **Clone and setup**:
```bash
git clone <your-repo>
cd python_backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download required models**:
```bash
python -c "from transformers import pipeline; pipeline('text-classification', 'Hello-SimpleAI/chatgpt-detector-roberta')"
```

3. **Run the server**:
```bash
python app.py
```

### Docker Deployment

1. **Create Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

2. **Build and run**:
```bash
docker build -t ai-detector-backend .
docker run -p 5000:5000 ai-detector-backend
```

### Cloud Deployment Options

#### 1. Heroku
```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create your-ai-detector
git push heroku main
```

#### 2. Google Cloud Run
```bash
gcloud run deploy ai-detector \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### 3. AWS Lambda (Serverless)
Use AWS Lambda with the Serverless framework for cost-effective deployment.

## API Endpoints

### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### 2. Analyze Single URL
```bash
POST /analyze
Content-Type: application/json

{
  "url": "https://instagram.com/p/ABC123/"
}
```

Response:
```json
{
  "success": true,
  "url": "https://instagram.com/p/ABC123/",
  "analysis": {
    "is_ai": false,
    "confidence": 87.5,
    "ai_probability": 12.5,
    "details": "High confidence human content (87.5%): Natural language patterns detected...",
    "features": {
      "avg_word_length": 4.2,
      "lexical_diversity": 0.78,
      "repetition_score": 0.15
    },
    "metadata": {
      "likes": 1250,
      "comments": 45,
      "date": "2024-01-15T10:30:00"
    },
    "processing_time": 1250
  }
}
```

### 3. Batch Analysis
```bash
POST /batch-analyze
Content-Type: application/json

{
  "urls": [
    "https://instagram.com/p/ABC123/",
    "https://youtube.com/shorts/XYZ789"
  ]
}
```

## Integration with Frontend

Update your React frontend to use the Python backend:

```typescript
// In your AIDetector component
const analyzeAIContent = async (url: string): Promise<AnalysisResult> => {
  const response = await fetch('http://your-backend-url/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ url }),
  });
  
  const data = await response.json();
  
  if (!data.success) {
    throw new Error(data.error || 'Analysis failed');
  }
  
  return {
    confidence: data.analysis.confidence,
    isAI: data.analysis.is_ai,
    details: data.analysis.details,
    processingTime: data.analysis.processing_time
  };
};
```

## Custom Model Training

### 1. Collect Training Data
```bash
python data_collection.py
```

This will:
- Generate AI text samples using OpenAI API
- Collect human texts from Reddit, news sources
- Create balanced training dataset

### 2. Train Custom Model
```bash
python train_model.py
```

This will:
- Load your training data
- Fine-tune a RoBERTa model
- Save the trained model
- Evaluate performance metrics

### 3. Use Custom Model
Update `app.py` to use your custom model:
```python
model_name = "./trained_ai_detector"  # Path to your trained model
```

## Performance Optimization

### 1. Model Optimization
- Use quantization for faster inference
- Implement model caching
- Use GPU acceleration when available

### 2. Caching Strategy
```python
from functools import lru_cache
import redis

# In-memory caching
@lru_cache(maxsize=1000)
def cached_analysis(text_hash):
    return analyze_content(text)

# Redis caching for production
redis_client = redis.Redis(host='localhost', port=6379, db=0)
```

### 3. Batch Processing
Process multiple URLs efficiently:
```python
# Use threading for I/O bound operations
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(extract_content, url) for url in urls]
    results = [future.result() for future in futures]
```

## Monitoring and Logging

### 1. Add Logging
```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log analysis requests
@app.before_request
def log_request():
    logger.info(f"Request: {request.method} {request.url}")
```

### 2. Add Metrics
```python
from prometheus_client import Counter, Histogram, generate_latest

# Track request metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')
```

## Security Considerations

### 1. Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_content():
    # Your analysis code
```

### 2. Input Validation
```python
from urllib.parse import urlparse

def validate_url(url):
    parsed = urlparse(url)
    allowed_domains = ['instagram.com', 'youtube.com', 'youtu.be']
    return any(domain in parsed.netloc for domain in allowed_domains)
```

## Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_openai_key
FLASK_ENV=production
MODEL_CACHE_DIR=/tmp/models
REDIS_URL=redis://localhost:6379
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure sufficient memory (8GB+ recommended)
   - Check internet connection for model downloads
   - Verify CUDA installation for GPU usage

2. **Content Extraction Failures**:
   - Instagram: May require session authentication
   - YouTube: API rate limits may apply
   - Use proxies for large-scale extraction

3. **Performance Issues**:
   - Enable model quantization
   - Use model caching
   - Implement request batching

### Debug Mode
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

## Production Checklist

- [ ] Set up proper logging
- [ ] Configure rate limiting
- [ ] Enable HTTPS
- [ ] Set up monitoring
- [ ] Configure auto-scaling
- [ ] Set up backup for trained models
- [ ] Implement proper error handling
- [ ] Add API documentation
- [ ] Set up CI/CD pipeline
- [ ] Configure environment variables

## Support

For issues and questions:
1. Check the logs for error details
2. Verify model compatibility
3. Test with sample data first
4. Monitor resource usage

## License
This project is licensed under the MIT License.