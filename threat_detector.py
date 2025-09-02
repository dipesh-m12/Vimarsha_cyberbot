import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
import json

# Model: BERT + LSTM + Classifier (must match training script)
class BertLSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 256, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256*2, num_classes)  # Bidirectional

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        lstm_output, _ = self.lstm(sequence_output)  # (batch, seq_len, 512)
        pooled = lstm_output[:, -1, :]  # Last hidden state
        dropped = self.dropout(pooled)
        return self.fc(dropped)

# Category labels
categories = ["safe", "sexual_offense", "hate_speech", "terrorism", "illicit_drugs", "scam", "violence"]

# Load tokenizer and model once (outside function)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained("threat_tokenizer")
model = BertLSTMClassifier(num_classes=7).to(device)
model.load_state_dict(torch.load("threat_classifier.pth"))
model.eval()

def classify_threat(texts, max_len=128):
    """
    Classify one or more texts for threats, returning results in JSON format.
    
    Args:
        texts (str or list): Single text string or list of text strings to classify.
        max_len (int): Maximum sequence length for tokenization (default: 128).
    
    Returns:
        str: JSON string with predictions and confidence scores for each text.
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    results = []
    for text in texts:
        # Tokenize input
        encoding = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_label = np.argmax(probabilities)
        
        # Format confidence scores as a dictionary
        confidence_scores = {cat: float(prob) for cat, prob in zip(categories, probabilities)}
        
        # Create result for this text
        result = {
            "text": text,
            "predicted": categories[pred_label],
            "confidence_scores": confidence_scores
        }
        results.append(result)
    
    # Return JSON string
    return json.dumps(results, indent=2)

# Sample usage (for testing only; can be removed when importing)
if __name__ == "__main__":
    test_texts = [
        "Nigger Motherfucker",
    ]
    print("Testing model on sample inputs:\n")
    print(classify_threat(test_texts))