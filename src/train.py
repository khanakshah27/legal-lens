from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def prepare_data(texts, labels):
    """Tokenize text for the Transformer model."""
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return encodings

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
