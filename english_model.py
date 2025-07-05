import pandas as pd
import numpy as np
from sklearn.utils import resample
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# 📊 Load and preprocess dataset
print("📊 Loading dataset...")
raw_dataset = load_dataset("boltuix/emotions-dataset")
train_data = raw_dataset["train"]

# Normalize labels to lowercase
def normalize_label(example):
    example["Label"] = example["Label"].lower()
    return example

train_data = train_data.map(normalize_label)

# Filter supported emotions
supported_emotions = {"anger", "joy", "sadness", "fear", "love", "surprise"}
filtered = train_data.filter(lambda x: x["Label"] in supported_emotions)
df = pd.DataFrame(filtered)

# 🏷 Check available labels
print("🔍 Available labels in dataset:", df["Label"].unique())

# ➕ Add 'happy' class manually from 'love' if 'joy' is unavailable
source_label = "love"  # you can change this to any available label
if source_label in df["Label"].values:
    print(f"✨ Adding 'happy' class using samples from '{source_label}'...")
    source_df = df[df["Label"] == source_label]
    happy_df = source_df.sample(n=500, replace=True, random_state=42)
    happy_df["Label"] = "happy"
    df = pd.concat([df, happy_df], ignore_index=True)
else:
    print(f"⚠️ Label '{source_label}' not found. Cannot create 'happy' class.")


# ⚖️ Balance dataset to the smallest class
print("🔁 Balancing dataset...")
min_count = df["Label"].value_counts().min()
balanced_df = pd.concat([
    resample(df[df["Label"] == label], replace=True, n_samples=min_count, random_state=42)
    for label in df["Label"].unique()
])
print("✅ Final label counts:\n", balanced_df["Label"].value_counts())

# 🔠 Label encoding
label2id = {label: idx for idx, label in enumerate(sorted(balanced_df["Label"].unique()))}
id2label = {idx: label for label, idx in label2id.items()}
balanced_df["label"] = balanced_df["Label"].map(label2id)

# 📦 Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(balanced_df)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = dataset["train"], dataset["test"]

# 🔠 Tokenizer
checkpoint = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize(example):
    return tokenizer(example["Sentence"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize)
val_ds = val_ds.map(tokenize)

# 🧠 Load Model
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ⚙️ Training Arguments
training_args = TrainingArguments(
    output_dir="./emotion-music-model",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none"
)

# 🏋️ Train without evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    
    tokenizer=tokenizer
)

print("🚀 Training started...")
trainer.train()

# 💾 Save the model and tokenizer
trainer.save_model("./emotion-music-model")
tokenizer.save_pretrained("./emotion-music-model")
print("✅ Model training complete and saved.")
