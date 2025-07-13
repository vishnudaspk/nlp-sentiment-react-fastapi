import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# --- CLI Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/data.jsonl", help="Path to JSONL dataset")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
args = parser.parse_args()

# --- Device Check ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Training on: {device.upper()}")

# --- Label Mapping ---
label2id = {"positive": 1, "negative": 0}
id2label = {v: k for k, v in label2id.items()}

# --- Load Dataset ---
dataset = load_dataset("json", data_files=args.data, split="train")
dataset = dataset.map(lambda e: {"label": label2id[e["label"]]})
print("✅ Dataset loaded and labels mapped.")

# --- Tokenization ---
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize(example):
    return tokenizer(example["text"], truncation=True)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")
print("✅ Tokenization complete.")

# --- Load Pretrained Model ---
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
).to(device)
print("✅ Model loaded and moved to GPU.")

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./model",
    learning_rate=args.lr,
    per_device_train_batch_size=8,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir="./model/logs",
    logging_steps=10,
    report_to="none"  # Disable wandb/other logging if not configured
)

# --- Trainer Setup ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# --- Train & Save Model ---
trainer.train()
trainer.save_model("./model")
tokenizer.save_pretrained("./model")
model.config.save_pretrained("./model")
print("✅ Fine-tuning complete. Model saved to ./model.")
