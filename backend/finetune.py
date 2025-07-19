#finetune.py

import os
import argparse
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    get_scheduler,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- CLI Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/data.jsonl")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--accum_steps", type=int, default=1)
args = parser.parse_args()

# --- Environment Setup ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Device:", device)

# --- Seed ---
set_seed(42)

# --- Label Mapping ---
label2id = {"positive": 0, "negative": 1}
id2label = {v: k for k, v in label2id.items()}

# --- Load Dataset ---
ds_raw = load_dataset("json", data_files=args.data, split="train")
ds_raw = ds_raw.filter(lambda ex: ex["label"] in label2id)
ds_raw = ds_raw.map(lambda ex: {"labels": label2id[ex["label"]]}, batched=False)

texts = list(ds_raw["text"])
labels = list(map(int, ds_raw["labels"]))

# --- Stratified Split ---
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# --- Create HF Datasets ---
def make_dataset(texts, labels):
    return Dataset.from_dict({"text": texts, "labels": labels})

splits = DatasetDict({
    "train": make_dataset(train_texts, train_labels),
    "validation": make_dataset(val_texts, val_labels),
    "test": make_dataset(test_texts, test_labels)
})
print("📊 Dataset sizes:", {k: len(v) for k, v in splits.items()})

# --- Tokenizer ---
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_fn(x):
    return tokenizer(x["text"], truncation=True, max_length=256)

splits = splits.map(tokenize_fn, batched=True)
splits.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# --- Model ---
model = BertForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(device)

model.gradient_checkpointing_enable()

# --- Class Weights for Imbalanced Data ---
class_counts = Counter(train_labels)
total = sum(class_counts.values())
weights = torch.tensor([total / class_counts[i] for i in range(len(label2id))], dtype=torch.float32).to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# --- Training Args ---
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    logging_dir="./model/logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=args.accum_steps,
    report_to="none"
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=splits["train"],
    eval_dataset=splits["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# --- Training ---
trainer.train()
print("✅ Validation metrics:", trainer.evaluate())
print("✅ Test metrics:", trainer.evaluate(eval_dataset=splits["test"]))

# --- Save ---
trainer.save_model("./model")
tokenizer.save_pretrained("./model")
print("✅ Model and tokenizer saved to ./model")

# --- Results Directory ---
os.makedirs("results", exist_ok=True)

# --- Save Metrics ---
val_metrics = trainer.evaluate(eval_dataset=splits["validation"])
test_metrics = trainer.evaluate(eval_dataset=splits["test"])

with open("results/val_metrics.json", "w") as f:
    json.dump(val_metrics, f, indent=4)

with open("results/test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=4)

# --- Classification Report ---
preds_output = trainer.predict(splits["test"])
y_true = preds_output.label_ids
y_pred = preds_output.predictions.argmax(-1)

report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(label2id))], output_dict=True)
with open("results/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label2id.keys(), yticklabels=label2id.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

print("📊 Results saved to 'results/' folder.")