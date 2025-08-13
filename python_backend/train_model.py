"""
Custom AI Detection Model Training Script
Train a model specifically for your use case with your own dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDetectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class AIDetectionModel(nn.Module):
    def __init__(self, model_name='roberta-base', num_classes=2, dropout=0.3):
        super(AIDetectionModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class AIDetectionTrainer:
    def __init__(self, model_name='roberta-base', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        
    def load_data(self, data_path):
        """Load training data from various sources"""
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Assume columns are 'text' and 'is_ai' (1 for AI, 0 for human)
        texts = df['text'].tolist()
        labels = df['is_ai'].tolist()
        
        return texts, labels
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        # Sample AI-generated texts (you would replace with real data)
        ai_texts = [
            "This is a comprehensive analysis of the current market trends that demonstrates significant opportunities for growth and development in various sectors.",
            "The implementation of advanced technologies has revolutionized the way businesses operate in today's digital landscape.",
            "Through careful consideration of multiple factors, we can conclude that the optimal solution involves a systematic approach to problem-solving.",
            "It is important to note that the effectiveness of this methodology depends on various parameters and environmental conditions.",
            "The research findings indicate a strong correlation between user engagement and platform optimization strategies."
        ] * 100  # Repeat to create more samples
        
        # Sample human-generated texts (you would replace with real data)
        human_texts = [
            "I can't believe how amazing that concert was last night! The energy was incredible.",
            "Just finished my morning run. Feeling great but man, those hills kicked my butt!",
            "Anyone know a good recipe for banana bread? I have some overripe bananas to use up.",
            "My dog keeps stealing my socks. It's cute but also annoying when I'm getting ready for work.",
            "This coffee shop has the worst WiFi ever. Can barely load my emails."
        ] * 100  # Repeat to create more samples
        
        texts = ai_texts + human_texts
        labels = [1] * len(ai_texts) + [0] * len(human_texts)
        
        # Shuffle the data
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return list(texts), list(labels)
    
    def prepare_data(self, texts, labels, test_size=0.2, val_size=0.1):
        """Split data into train/val/test sets"""
        # First split: train+val vs test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Second split: train vs val
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, 
            test_size=val_size/(1-test_size), random_state=42, stratify=train_val_labels
        )
        
        return {
            'train': (train_texts, train_labels),
            'val': (val_texts, val_labels),
            'test': (test_texts, test_labels)
        }
    
    def create_dataloaders(self, data_splits, batch_size=16):
        """Create PyTorch DataLoaders"""
        dataloaders = {}
        
        for split_name, (texts, labels) in data_splits.items():
            dataset = AIDetectionDataset(texts, labels, self.tokenizer)
            shuffle = split_name == 'train'
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            dataloaders[split_name] = dataloader
        
        return dataloaders
    
    def train_model(self, dataloaders, num_epochs=5, learning_rate=2e-5):
        """Train the AI detection model"""
        self.model = AIDetectionModel(self.model_name).to(self.device)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_training_steps = num_epochs * len(dataloaders['train'])
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            train_predictions = []
            train_labels = []
            
            for batch in tqdm(dataloaders['train'], desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                train_predictions.extend(predictions.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(dataloaders['val'], desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_acc = accuracy_score(train_labels, train_predictions)
            val_acc = accuracy_score(val_labels, val_predictions)
            
            logger.info(f"Train Loss: {total_loss/len(dataloaders['train']):.4f}, "
                       f"Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss/len(dataloaders['val']):.4f}, "
                       f"Val Acc: {val_acc:.4f}")
    
    def evaluate_model(self, dataloader):
        """Evaluate the trained model"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                batch_predictions = torch.argmax(outputs, dim=1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, save_path):
        """Save the trained model and tokenizer"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save model config
        config = {
            'model_name': self.model_name,
            'num_classes': 2,
            'max_length': 512
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        # Load config
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        self.model = AIDetectionModel(config['model_name']).to(self.device)
        self.model.load_state_dict(torch.load(
            os.path.join(model_path, 'model.pt'), 
            map_location=self.device
        ))
        
        logger.info(f"Model loaded from {model_path}")

def main():
    """Main training pipeline"""
    trainer = AIDetectionTrainer()
    
    # Option 1: Load from file (uncomment if you have data)
    # texts, labels = trainer.load_data('your_dataset.csv')
    
    # Option 2: Use sample data for demonstration
    texts, labels = trainer.create_sample_dataset()
    
    logger.info(f"Loaded {len(texts)} samples")
    
    # Prepare data splits
    data_splits = trainer.prepare_data(texts, labels)
    logger.info(f"Train: {len(data_splits['train'][0])}, "
               f"Val: {len(data_splits['val'][0])}, "
               f"Test: {len(data_splits['test'][0])}")
    
    # Create dataloaders
    dataloaders = trainer.create_dataloaders(data_splits, batch_size=16)
    
    # Train model
    trainer.train_model(dataloaders, num_epochs=3, learning_rate=2e-5)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_model(dataloaders['test'])
    logger.info(f"Test Results: {test_metrics}")
    
    # Save model
    trainer.save_model('./trained_ai_detector')

if __name__ == "__main__":
    main()