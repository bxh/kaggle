# %% [markdown]
# # Import Libraries

# %% [code] {"execution":{"iopub.status.busy":"2024-10-16T04:03:19.206428Z","iopub.execute_input":"2024-10-16T04:03:19.207068Z","iopub.status.idle":"2024-10-16T04:03:25.729468Z","shell.execute_reply.started":"2024-10-16T04:03:19.207029Z","shell.execute_reply":"2024-10-16T04:03:25.728573Z"}}
# Data manipulation
import pandas as pd
import numpy as np

# Text processing
import re
import nltk
from nltk.corpus import stopwords

# Machine Learning
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Transformers for BERT embeddings
from transformers import BertTokenizerFast, BertModel

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from IPython.display import display

import ast

# %% [markdown]
# # Train

# %% [markdown]
# ## Load data

# %% [code] {"execution":{"iopub.status.busy":"2024-10-16T05:17:06.441483Z","iopub.execute_input":"2024-10-16T05:17:06.441871Z","iopub.status.idle":"2024-10-16T05:17:06.812916Z","shell.execute_reply.started":"2024-10-16T05:17:06.441834Z","shell.execute_reply":"2024-10-16T05:17:06.812063Z"}}
train_df = pd.read_csv('/kaggle/input/nbme-score-clinical-patient-notes/train.csv')

test_df = pd.read_csv('/kaggle/input/nbme-score-clinical-patient-notes/test.csv')

patient_notes_df = pd.read_csv('/kaggle/input/nbme-score-clinical-patient-notes/patient_notes.csv')

features_df = pd.read_csv('/kaggle/input/nbme-score-clinical-patient-notes/features.csv')

# %% [markdown]
# ## EDA

# %% [code] {"execution":{"iopub.status.busy":"2024-10-16T05:17:03.469358Z","iopub.execute_input":"2024-10-16T05:17:03.470218Z","iopub.status.idle":"2024-10-16T05:17:03.501731Z","shell.execute_reply.started":"2024-10-16T05:17:03.470182Z","shell.execute_reply":"2024-10-16T05:17:03.500839Z"}}
print("train_df")
display(train_df.head())

print("test_df")
display(test_df.head())

print("patient_notes_df")
display(patient_notes_df.head())

print("features_df")
display(features_df.head())



# %% [code] {"execution":{"iopub.status.busy":"2024-10-16T05:17:09.962067Z","iopub.execute_input":"2024-10-16T05:17:09.962461Z","iopub.status.idle":"2024-10-16T05:17:10.355367Z","shell.execute_reply.started":"2024-10-16T05:17:09.962422Z","shell.execute_reply":"2024-10-16T05:17:10.354291Z"}}
patient_notes_df_subset = patient_notes_df[['pn_num', 'pn_history']]
features_df_subset = features_df[['feature_num', 'feature_text']]

train_merged_df = train_df.merge(patient_notes_df_subset, on='pn_num', how='left')
train_merged_df = train_merged_df.merge(features_df_subset, on='feature_num', how='left')
train_merged_df['annotation'] = train_merged_df['annotation'].apply(ast.literal_eval)
train_merged_df['location'] = train_merged_df['location'].apply(ast.literal_eval)
print("train_merged_df")
display(train_merged_df.head())

test_merged_df = test_df.merge(patient_notes_df_subset, on='pn_num', how='left')
test_merged_df = test_merged_df.merge(features_df_subset, on='feature_num', how='left')
print("test_merged_df")
display(test_merged_df.head())

# %% [markdown]
# ## Create Token Labels

# %% [code] {"execution":{"iopub.status.busy":"2024-10-16T05:26:42.967736Z","iopub.execute_input":"2024-10-16T05:26:42.968504Z","iopub.status.idle":"2024-10-16T05:26:43.188541Z","shell.execute_reply.started":"2024-10-16T05:26:42.968465Z","shell.execute_reply":"2024-10-16T05:26:43.187581Z"}}
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def create_token_labels(row, mode='train'):
    """
    Function to tokenize the text and assign labels for training or just tokenize for test mode.
    
    Args:
        row (pd.Series): A row of the DataFrame containing text and annotations (for training mode).
        mode (str): Either 'train' or 'test'. In 'train' mode, the function will create token labels,
                    while in 'test' mode, it will only tokenize the text and return 'O' for all labels.
    
    Returns:
        tokens (list): List of tokens generated from the tokenizer.
        token_labels (list): List of token labels ('O' for test mode, actual labels for train mode).
    """
    text = row['pn_history']

    # Tokenize the text with offset mapping (for character-level alignment)
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offset_mapping = encoding['offset_mapping']

    # If in test mode, return tokens and 'O' labels for all tokens (since no annotations are available)
    if mode == 'test':
        token_labels = ['O'] * len(tokens)
        return tokens, token_labels

    # In train mode, create token labels based on the provided annotations and locations
    annotations = row['annotation']
    locations = row['location']
    feature_num = row['feature_num']

    # Initialize labels as 'O' for all tokens
    token_labels = ['O'] * len(tokens)
    
    # Generate token labels for each annotation based on the provided locations
    for annotation, annotation_locations in zip(annotations, locations):
        for location in annotation_locations.split(';'):
            start_char, end_char = map(int, location.split())

            # Iterate through the token offsets and assign labels
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start >= start_char and token_end <= end_char:
                    if token_labels[i] == 'O':  # If this token has not already been labeled
                        token_labels[i] = f'B-feature-{feature_num}' if token_start == start_char else f'I-feature-{feature_num}'

    return tokens, token_labels

# %% [code] {"execution":{"iopub.status.busy":"2024-10-16T05:27:07.364034Z","iopub.execute_input":"2024-10-16T05:27:07.364685Z","iopub.status.idle":"2024-10-16T05:27:24.282444Z","shell.execute_reply.started":"2024-10-16T05:27:07.364643Z","shell.execute_reply":"2024-10-16T05:27:24.281432Z"}}
train_merged_df[['tokens', 'labels']] = train_merged_df.apply(lambda row: pd.Series(create_token_labels(row, mode='train')), axis=1)
display(train_merged_df.head())

test_merged_df[['tokens', 'labels']] = test_merged_df.apply(lambda row: pd.Series(create_token_labels(row, mode='test')), axis=1)
display(test_merged_df.head())

# %% [markdown]
# ## Create Data Loader

# %% [code] {"execution":{"iopub.status.busy":"2024-10-16T05:48:47.197998Z","iopub.execute_input":"2024-10-16T05:48:47.199065Z","iopub.status.idle":"2024-10-16T05:48:47.547340Z","shell.execute_reply.started":"2024-10-16T05:48:47.199022Z","shell.execute_reply":"2024-10-16T05:48:47.546321Z"}}
from torch.utils.data import Dataset

class ClinicalNotesTokenDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        
        # Tokenize the text and prepare input
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        if self.labels is None:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }
        
        labels = self.labels[index]
        word_ids = encoding.word_ids()
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens ([CLS], [SEP], or padding)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])  # Assign the label
            else:
                label_ids.append(labels[word_idx])  # Inside the same word

            previous_word_idx = word_idx

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

# Convert the tokens and labels columns into lists
texts = train_merged_df['tokens'].tolist()

# Maximum sequence length (you can adjust this based on your use case)
max_len = 128

unique_labels = train_merged_df['labels'].explode().unique()
label_map = {label: idx for idx, label in enumerate(unique_labels)}

def map_labels_to_indices(labels):
    return [label_map[label] for label in labels]

train_merged_df['label_indices'] = train_merged_df['labels'].apply(map_labels_to_indices)

label_indices = train_merged_df['label_indices'].tolist()

train_dataset = ClinicalNotesTokenDataset(
    texts=texts,
    labels=label_indices,
    tokenizer=tokenizer,
    max_len=max_len
)

# Create the DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-16T05:48:57.806807Z","iopub.execute_input":"2024-10-16T05:48:57.807455Z","iopub.status.idle":"2024-10-16T05:48:57.812607Z","shell.execute_reply.started":"2024-10-16T05:48:57.807413Z","shell.execute_reply":"2024-10-16T05:48:57.811598Z"}}
test_texts = test_merged_df['tokens'].tolist()

test_dataset = ClinicalNotesTokenDataset(
    texts=test_texts,
    labels=None,    
    tokenizer=tokenizer,
    max_len=max_len
)

# Create the DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# %% [markdown]
# ## Train Loop

# %% [code] {"execution":{"iopub.status.busy":"2024-10-16T04:05:32.987486Z","iopub.execute_input":"2024-10-16T04:05:32.987866Z","iopub.status.idle":"2024-10-16T04:05:37.393839Z","shell.execute_reply.started":"2024-10-16T04:05:32.987827Z","shell.execute_reply":"2024-10-16T04:05:37.392791Z"}}
import os
from transformers import BertForTokenClassification, AdamW
import torch
import torch.nn as nn

# Number of labels (including O, B/I with different features)
unique_labels = train_merged_df['labels'].explode().unique()
label_map = {label: idx for idx, label in enumerate(unique_labels)}
num_labels = len(unique_labels)

# Load pre-trained BERT model with a token classification head
model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_labels
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Checkpoint directory
checkpoint_dir = "/kaggle/working/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to save a checkpoint
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')

# Function to load from a checkpoint
def load_checkpoint(model, optimizer, checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f'Loaded checkpoint from {checkpoint_path} at epoch {start_epoch-1}')
        return start_epoch, loss
    return 0, float('inf')  # Return 0 if no checkpoint is found

# Load from the last checkpoint if available
start_epoch, previous_loss = load_checkpoint(model, optimizer, checkpoint_dir)

# Training loop
epochs = 10
for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0

    for batch in t:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    # Save a checkpoint every few epochs (e.g., every 2 epochs)
    if (epoch + 1) % 2 == 0:
        save_checkpoint(epoch, model, optimizer, avg_loss, checkpoint_dir)



# %% [markdown]
# ## Validation

# %% [code] {"execution":{"iopub.status.busy":"2024-10-16T05:56:42.538689Z","iopub.execute_input":"2024-10-16T05:56:42.539073Z","iopub.status.idle":"2024-10-16T05:56:42.616005Z","shell.execute_reply.started":"2024-10-16T05:56:42.539035Z","shell.execute_reply":"2024-10-16T05:56:42.615101Z"}}
from transformers import BertForTokenClassification
import torch

model.eval()  # Set the model to evaluation mode

# Define a reverse mapping from index to label
id2label = {v: k for k, v in label_map.items()}  # Convert indices back to label names

# Inference function for test data using predefined variables
def predict_test_labels(model, test_loader):
    model.eval()  # Ensure the model is in evaluation mode
    predictions = []
    
    with torch.no_grad():  # Disable gradient computation during inference
        for batch in test_loader:
            # Move input tensors to the appropriate device (GPU or CPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Logits shape: (batch_size, seq_len, num_labels)

            # Get the predicted class indices for each token (shape: batch_size, seq_len)
            predicted_indices = torch.argmax(logits, dim=-1).cpu().numpy()
            print(outputs)
            
            # Convert predicted indices to actual labels using the id2label mapping
            for preds in predicted_indices:
                pred_labels = [id2label[pred] for pred in preds]
                predictions.append(pred_labels)
    
    return predictions

# Call the inference function on the test_loader
test_predictions = predict_test_labels(model, test_loader)

# Optionally: Display a sample of the predictions
for i, preds in enumerate(test_predictions[:5]):  # Display the first 5 samples
    print(f"Sample {i + 1}:")
    print(f"Predicted Labels: {preds}")


