import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # Import AdamW from torch.optim
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm

# Custom Dataset class
class ThreatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Model: BERT + LSTM + Classifier
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

# Load data
df = pd.read_csv("threat_dataset.csv")
texts = df['text'].values
labels = df['label'].values
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, stratify=labels)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = ThreatDataset(train_texts, train_labels, tokenizer)
val_dataset = ThreatDataset(val_texts, val_labels, tokenizer)

# Tunable params
batch_size = 32  # Try 32
lr = 3e-5  # Try 3e-5
epochs = 5  # Increased
patience = 2  # For early stopping

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertLSTMClassifier(num_classes=7).to(device)
optimizer = AdamW(model.parameters(), lr=lr)  # Use torch.optim.AdamW
criterion = nn.CrossEntropyLoss()

# Add scheduler
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)  # Or warmup_steps=int(0.1 * total_steps)

# Training loop with early stopping
best_val_acc = 0
early_stop_count = 0
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update lr
        train_loss += loss.item()
    print(f"Epoch {epoch+1} Train Loss: {train_loss / len(train_loader)}")

    # Validation
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            true.extend(labels.cpu().numpy())
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average='weighted')  # Better for multi-class
    print(f"Validation Accuracy: {acc} | F1-Score: {f1}")

    # Early stopping
    if acc > best_val_acc:
        best_val_acc = acc
        early_stop_count = 0
        torch.save(model.state_dict(), "best_threat_classifier.pth")  # Save best
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("Early stopping triggered.")
            break

# Load best model for final save/use
model.load_state_dict(torch.load("best_threat_classifier.pth"))
torch.save(model.state_dict(), "threat_classifier.pth")
tokenizer.save_pretrained("threat_tokenizer")
print("Model trained and saved!")