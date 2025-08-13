from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from typing import Dict, Any
import os
from urllib.parse import urlparse
import yt_dlp
import instaloader

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIContentDetector:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models for AI detection"""
        try:
            # Primary AI detection model
            model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
            self.models['ai_detector'] = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Backup model for comparison
            backup_model = "roberta-base-openai-detector"
            self.models['backup_detector'] = pipeline(
                "text-classification",
                model="roberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models = {}
    
    def extract_instagram_content(self, url: str) -> Dict[str, Any]:
        """Extract content from Instagram posts"""
        try:
            L = instaloader.Instaloader()
            
            # Extract shortcode from URL
            shortcode = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]
            
            # Get post
            post = instaloader.Post.from_shortcode(L.context, shortcode)
            
            content = {
                'text': post.caption or '',
                'metadata': {
                    'likes': post.likes,
                    'comments': post.comments,
                    'date': post.date.isoformat(),
                    'is_video': post.is_video,
                    'owner': post.owner_username
                }
            }
            
            return content
        except Exception as e:
            logger.error(f"Error extracting Instagram content: {e}")
            return {'text': '', 'metadata': {}}
    
    def extract_youtube_content(self, url: str) -> Dict[str, Any]:
        """Extract content from YouTube Shorts"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                content = {
                    'text': f"{info.get('title', '')} {info.get('description', '')}",
                    'metadata': {
                        'title': info.get('title', ''),
                        'description': info.get('description', ''),
                        'view_count': info.get('view_count', 0),
                        'like_count': info.get('like_count', 0),
                        'duration': info.get('duration', 0),
                        'upload_date': info.get('upload_date', ''),
                        'uploader': info.get('uploader', '')
                    }
                }
                
                return content
        except Exception as e:
            logger.error(f"Error extracting YouTube content: {e}")
            return {'text': '', 'metadata': {}}
    
    def extract_content_from_url(self, url: str) -> Dict[str, Any]:
        """Extract content based on URL type"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        if 'instagram.com' in domain:
            return self.extract_instagram_content(url)
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            return self.extract_youtube_content(url)
        else:
            # Generic web scraping fallback
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(strip=True)
                return {'text': text[:2000], 'metadata': {'source': 'web_scraping'}}
            except Exception as e:
                logger.error(f"Error with generic extraction: {e}")
                return {'text': '', 'metadata': {}}
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        # Limit length for model processing
        return text[:512]
    
    def calculate_linguistic_features(self, text: str) -> Dict[str, float]:
        """Calculate linguistic features for analysis"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        features = {
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences if sent.strip()]) if sentences else 0,
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'punctuation_ratio': len(re.findall(r'[.,!?;:]', text)) / len(text) if text else 0,
            'capitalization_ratio': len(re.findall(r'[A-Z]', text)) / len(text) if text else 0,
            'repetition_score': 1 - (len(set(words)) / len(words)) if words else 0
        }
        
        return features
    
    def ensemble_prediction(self, text: str) -> Dict[str, Any]:
        """Use ensemble of models for more accurate prediction"""
        predictions = []
        
        # Primary model prediction
        if 'ai_detector' in self.models:
            try:
                result = self.models['ai_detector'](text)
                if isinstance(result, list) and len(result) > 0:
                    # Find AI/Human labels
                    ai_score = 0
                    for item in result:
                        if item['label'] in ['LABEL_1', 'AI', 'MACHINE']:
                            ai_score = item['score']
                        elif item['label'] in ['LABEL_0', 'HUMAN']:
                            ai_score = 1 - item['score']
                    predictions.append(ai_score)
            except Exception as e:
                logger.error(f"Primary model error: {e}")
        
        # Linguistic features analysis
        features = self.calculate_linguistic_features(text)
        
        # Heuristic scoring based on features
        heuristic_score = 0
        if features['repetition_score'] > 0.3:
            heuristic_score += 0.3
        if features['lexical_diversity'] < 0.4:
            heuristic_score += 0.2
        if features['avg_word_length'] > 6:
            heuristic_score += 0.2
        if features['avg_sentence_length'] > 20:
            heuristic_score += 0.3
        
        predictions.append(heuristic_score)
        
        # Ensemble average
        final_score = np.mean(predictions) if predictions else 0.5
        
        return {
            'ai_probability': final_score,
            'confidence': abs(final_score - 0.5) * 2,  # Distance from uncertain (0.5)
            'features': features,
            'model_predictions': predictions
        }
    
    def analyze_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis function"""
        start_time = time.time()
        
        text = content.get('text', '')
        if not text.strip():
            return {
                'error': 'No text content found to analyze',
                'processing_time': time.time() - start_time
            }
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get ensemble prediction
        prediction = self.ensemble_prediction(processed_text)
        
        ai_probability = prediction['ai_probability']
        confidence = prediction['confidence']
        is_ai = ai_probability > 0.5
        
        # Generate detailed analysis
        if is_ai:
            if confidence > 0.8:
                details = f"High confidence AI detection ({confidence*100:.1f}%): Strong indicators of machine-generated patterns detected."
            elif confidence > 0.6:
                details = f"Moderate confidence AI detection ({confidence*100:.1f}%): Some patterns suggest AI generation."
            else:
                details = f"Low confidence AI detection ({confidence*100:.1f}%): Mixed signals detected."
        else:
            if confidence > 0.8:
                details = f"High confidence human content ({confidence*100:.1f}%): Natural language patterns detected."
            elif confidence > 0.6:
                details = f"Moderate confidence human content ({confidence*100:.1f}%): Mostly human characteristics."
            else:
                details = f"Low confidence human content ({confidence*100:.1f}%): Uncertain classification."
        
        # Add feature analysis to details
        features = prediction['features']
        details += f" Lexical diversity: {features['lexical_diversity']:.2f}, "
        details += f"Avg word length: {features['avg_word_length']:.1f}, "
        details += f"Repetition score: {features['repetition_score']:.2f}."
        
        return {
            'is_ai': is_ai,
            'confidence': confidence * 100,
            'ai_probability': ai_probability * 100,
            'details': details,
            'features': features,
            'metadata': content.get('metadata', {}),
            'processing_time': (time.time() - start_time) * 1000  # Convert to ms
        }

# Initialize detector
detector = AIContentDetector()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': len(detector.models) > 0})

@app.route('/analyze', methods=['POST'])
def analyze_content():
    """Main analysis endpoint"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Extract content from URL
        content = detector.extract_content_from_url(url)
        
        if not content.get('text'):
            return jsonify({'error': 'Could not extract content from URL'}), 400
        
        # Analyze content
        result = detector.analyze_content(content)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            'url': url,
            'analysis': result
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Batch analysis endpoint for multiple URLs"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls or len(urls) > 10:  # Limit batch size
            return jsonify({'error': 'Provide 1-10 URLs for batch analysis'}), 400
        
        results = []
        for url in urls:
            try:
                content = detector.extract_content_from_url(url)
                if content.get('text'):
                    analysis = detector.analyze_content(content)
                    results.append({
                        'url': url,
                        'success': 'error' not in analysis,
                        'analysis': analysis
                    })
                else:
                    results.append({
                        'url': url,
                        'success': False,
                        'error': 'Could not extract content'
                    })
            except Exception as e:
                results.append({
                    'url': url,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)