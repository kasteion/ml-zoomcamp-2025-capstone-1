# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# Model parameters
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.0,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_dir="./logs"
)

# Output File

# Load data
df = pd.read_csv('data/sentiment_analysis.csv')

# Prepare data
df = df[['text', 'sentiment']]

label_encoder = LabelEncoder()
label_encoder.fit(df.sentiment)

df.sentiment = label_encoder.transform(df.sentiment)
X = df.text
y = df.sentiment

# Split dataset
df_full_train, df_test, y_full_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df_train, df_val, y_train, y_val = train_test_split(df_full_train, y_full_train, test_size=0.25, random_state=42)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

train_encodings = tokenize(df_train.to_list())
val_encodings = tokenize(df_val.to_list())
test_encodings = tokenize(df_test.to_list())

train_dataset = SentimentDataset(train_encodings, y_train.tolist())
val_dataset = SentimentDataset(val_encodings, y_val.tolist())
test_dataset = SentimentDataset(test_encodings, y_test.tolist()) 


# Train model
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

finetuned_distilbert = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=3
)
trainer = Trainer(
    model=finetuned_distilbert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# Export model
finetuned_distilbert.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")