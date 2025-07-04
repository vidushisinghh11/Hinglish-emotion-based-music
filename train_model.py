import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("📊 Loading dataset...")

# ✅ Load and rename columns
df = pd.read_csv("data/data.csv")
df = df.rename(columns={"Statements": "text", "category": "label"})
df = df[["text", "label"]]  # Keep only required columns

# 🔍 Print original class distribution
print("Original class distribution:")
print(df["label"].value_counts())

# ⚖️ Balance dataset by upsampling minority classes
min_class_count = df["label"].value_counts().min()
balanced_df = pd.concat([
    resample(df[df["label"] == label],
             replace=True,
             n_samples=min_class_count,
             random_state=42)
    for label in df["label"].unique()
])

# ✅ Confirm new balanced distribution
print("\nBalanced class distribution:")
print(balanced_df["label"].value_counts())

# 🔠 Label encoding
unique_labels = sorted(balanced_df["label"].unique())
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label = {i: l for l, i in label2id.items()}
balanced_df["label"] = balanced_df["label"].map(label2id)

# ✅ Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(balanced_df)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
val_ds = dataset["test"]

# 🔠 Tokenizing
print("🔠 Tokenizing...")
checkpoint = "l3cube-pune/hing-roberta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# 🧠 Load model
print("🧠 Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 📈 Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ⚙️ Training settings
training_args = TrainingArguments(
    output_dir="./hing-emotion-model",
    per_device_train_batch_size=16,
    num_train_epochs=4,
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=False  # change to True if adding evaluation
)

# 🚀 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# 🔁 Train model
print("\n🚀 Training started...")
trainer.train()

# 💾 Save model and tokenizer
trainer.save_model("./hing-emotion-model")
tokenizer.save_pretrained("./hing-emotion-model")
print("\n✅ Model and tokenizer saved to ./hing-emotion-model")
