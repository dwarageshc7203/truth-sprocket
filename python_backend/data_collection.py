"""
Data Collection Script for AI Detection Training
Collect and prepare datasets from various sources
"""

import requests
import pandas as pd
import numpy as np
import time
import json
import os
from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai  # For generating AI samples
from datasets import load_dataset
import instaloader
import yt_dlp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def collect_ai_generated_texts(self, num_samples=1000) -> List[str]:
        """Generate AI text samples using OpenAI API"""
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. Using placeholder data.")
            return self._generate_placeholder_ai_texts(num_samples)
        
        prompts = [
            "Write a social media post about technology trends",
            "Create a product review for a smartphone",
            "Write about climate change and its effects",
            "Describe a travel experience",
            "Write a fitness motivation post",
            "Create content about cooking tips",
            "Write about personal development",
            "Describe a business strategy",
            "Write about educational topics",
            "Create content about entertainment news"
        ]
        
        ai_texts = []
        
        for i in range(num_samples):
            try:
                prompt = random.choice(prompts)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a social media content creator. Write engaging, natural-sounding posts."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                
                text = response.choices[0].message.content.strip()
                ai_texts.append(text)
                
                if i % 100 == 0:
                    logger.info(f"Generated {i} AI texts")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error generating AI text {i}: {e}")
                continue
        
        return ai_texts
    
    def _generate_placeholder_ai_texts(self, num_samples) -> List[str]:
        """Generate placeholder AI-like texts for demonstration"""
        templates = [
            "This comprehensive analysis demonstrates significant potential for {topic} in today's dynamic marketplace.",
            "Through careful consideration of multiple factors, we can optimize {topic} for maximum efficiency.",
            "The implementation of advanced {topic} strategies has revolutionized industry standards.",
            "It is important to note that {topic} effectiveness depends on various environmental parameters.",
            "Research findings indicate strong correlations between {topic} and user engagement metrics."
        ]
        
        topics = [
            "technology integration", "digital transformation", "market optimization",
            "data analytics", "user experience", "content creation", "social engagement",
            "brand development", "strategic planning", "innovation management"
        ]
        
        ai_texts = []
        for _ in range(num_samples):
            template = random.choice(templates)
            topic = random.choice(topics)
            text = template.format(topic=topic)
            ai_texts.append(text)
        
        return ai_texts
    
    def collect_human_texts_from_reddit(self, subreddits, num_samples=1000) -> List[str]:
        """Collect human-written texts from Reddit"""
        human_texts = []
        
        headers = {
            'User-Agent': 'AIDetectionDataCollector/1.0'
        }
        
        for subreddit in subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=100"
                response = requests.get(url, headers=headers)
                data = response.json()
                
                for post in data['data']['children']:
                    post_data = post['data']
                    
                    # Collect post titles and text
                    if post_data.get('selftext') and len(post_data['selftext']) > 50:
                        human_texts.append(post_data['selftext'])
                    
                    if len(post_data.get('title', '')) > 30:
                        human_texts.append(post_data['title'])
                    
                    if len(human_texts) >= num_samples:
                        break
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit}: {e}")
                continue
        
        return human_texts[:num_samples]
    
    def collect_instagram_captions(self, usernames, max_posts_per_user=50) -> List[str]:
        """Collect Instagram captions (human-written)"""
        L = instaloader.Instaloader()
        human_texts = []
        
        for username in usernames:
            try:
                profile = instaloader.Profile.from_username(L.context, username)
                
                for i, post in enumerate(profile.get_posts()):
                    if i >= max_posts_per_user:
                        break
                    
                    if post.caption and len(post.caption) > 50:
                        human_texts.append(post.caption)
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting from @{username}: {e}")
                continue
        
        return human_texts
    
    def collect_youtube_descriptions(self, channel_urls, max_videos_per_channel=30) -> List[str]:
        """Collect YouTube video descriptions"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        human_texts = []
        
        for channel_url in channel_urls:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    channel_info = ydl.extract_info(channel_url, download=False)
                    
                    if 'entries' in channel_info:
                        for i, video in enumerate(channel_info['entries']):
                            if i >= max_videos_per_channel:
                                break
                            
                            try:
                                video_info = ydl.extract_info(
                                    f"https://youtube.com/watch?v={video['id']}", 
                                    download=False
                                )
                                
                                description = video_info.get('description', '')
                                if len(description) > 100:
                                    human_texts.append(description[:500])  # Limit length
                                
                            except Exception as e:
                                logger.error(f"Error processing video {video['id']}: {e}")
                                continue
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting from channel {channel_url}: {e}")
                continue
        
        return human_texts
    
    def collect_news_articles(self, news_sources, num_articles=500) -> List[str]:
        """Collect news articles (human-written)"""
        # This would use news APIs like NewsAPI
        # For demonstration, using placeholder data
        
        news_templates = [
            "Local authorities reported {event} in {location} yesterday evening.",
            "Breaking: {event} causes significant impact on {location} residents.",
            "Witnesses describe {event} as unprecedented in {location} history.",
            "Officials confirm {event} investigation ongoing in {location}.",
            "Community leaders respond to {event} affecting {location} area."
        ]
        
        events = ["power outage", "traffic incident", "weather alert", "community event", "local election"]
        locations = ["downtown", "suburban area", "city center", "residential district", "business quarter"]
        
        human_texts = []
        for _ in range(num_articles):
            template = random.choice(news_templates)
            event = random.choice(events)
            location = random.choice(locations)
            text = template.format(event=event, location=location)
            human_texts.append(text)
        
        return human_texts
    
    def load_existing_datasets(self) -> Dict[str, List[str]]:
        """Load existing AI detection datasets from Hugging Face"""
        datasets_info = {
            'ai_texts': [],
            'human_texts': []
        }
        
        try:
            # Load AI-generated text dataset
            dataset = load_dataset("Hello-SimpleAI/HC3", split="train[:1000]")
            for item in dataset:
                if 'chatgpt_answers' in item and item['chatgpt_answers']:
                    datasets_info['ai_texts'].extend(item['chatgpt_answers'])
                if 'human_answers' in item and item['human_answers']:
                    datasets_info['human_texts'].extend(item['human_answers'])
        
        except Exception as e:
            logger.error(f"Error loading HC3 dataset: {e}")
        
        return datasets_info
    
    def clean_and_filter_texts(self, texts: List[str], min_length=50, max_length=1000) -> List[str]:
        """Clean and filter collected texts"""
        cleaned_texts = []
        
        for text in texts:
            # Basic cleaning
            text = text.strip()
            text = ' '.join(text.split())  # Normalize whitespace
            
            # Filter by length
            if min_length <= len(text) <= max_length:
                # Remove texts with too many special characters
                special_char_ratio = sum(1 for c in text if not c.isalnum() and c != ' ') / len(text)
                if special_char_ratio < 0.3:
                    cleaned_texts.append(text)
        
        return cleaned_texts
    
    def create_balanced_dataset(self, ai_texts: List[str], human_texts: List[str]) -> pd.DataFrame:
        """Create a balanced dataset for training"""
        # Balance the dataset
        min_size = min(len(ai_texts), len(human_texts))
        
        ai_texts = ai_texts[:min_size]
        human_texts = human_texts[:min_size]
        
        # Create DataFrame
        data = []
        
        for text in ai_texts:
            data.append({'text': text, 'is_ai': 1, 'source': 'ai_generated'})
        
        for text in human_texts:
            data.append({'text': text, 'is_ai': 0, 'source': 'human_written'})
        
        df = pd.DataFrame(data)
        
        # Shuffle the dataset
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """Save the dataset to file"""
        df.to_csv(filename, index=False)
        logger.info(f"Dataset saved to {filename}")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"AI samples: {len(df[df['is_ai'] == 1])}")
        logger.info(f"Human samples: {len(df[df['is_ai'] == 0])}")

def main():
    """Main data collection pipeline"""
    collector = DataCollector(openai_api_key=os.getenv('OPENAI_API_KEY'))
    
    logger.info("Starting data collection...")
    
    # Collect AI-generated texts
    logger.info("Collecting AI-generated texts...")
    ai_texts = collector.collect_ai_generated_texts(num_samples=1000)
    
    # Collect human texts from various sources
    logger.info("Collecting human texts...")
    
    # Reddit
    subreddits = ['AskReddit', 'todayilearned', 'LifeProTips', 'explainlikeimfive']
    reddit_texts = collector.collect_human_texts_from_reddit(subreddits, num_samples=500)
    
    # News articles
    news_texts = collector.collect_news_articles(['example_news'], num_articles=300)
    
    # Load existing datasets
    existing_data = collector.load_existing_datasets()
    
    # Combine all human texts
    all_human_texts = reddit_texts + news_texts + existing_data['human_texts']
    all_ai_texts = ai_texts + existing_data['ai_texts']
    
    # Clean and filter
    logger.info("Cleaning and filtering texts...")
    clean_ai_texts = collector.clean_and_filter_texts(all_ai_texts)
    clean_human_texts = collector.clean_and_filter_texts(all_human_texts)
    
    logger.info(f"Clean AI texts: {len(clean_ai_texts)}")
    logger.info(f"Clean human texts: {len(clean_human_texts)}")
    
    # Create balanced dataset
    dataset = collector.create_balanced_dataset(clean_ai_texts, clean_human_texts)
    
    # Save dataset
    collector.save_dataset(dataset, 'ai_detection_dataset.csv')
    
    # Create smaller dataset for quick testing
    test_dataset = dataset.sample(n=min(1000, len(dataset)))
    collector.save_dataset(test_dataset, 'ai_detection_test_dataset.csv')

if __name__ == "__main__":
    main()