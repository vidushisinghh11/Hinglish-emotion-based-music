from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re

# ✅ Local model paths (relative to your working directory)
ENGLISH_MODEL_PATH = "english_music_model"
HINGLISH_MODEL_PATH = "hinglish_model/hing-emotion-model"

# 🔁 Load English model (local only)
print("🔁 Loading English model...")
tokenizer_en = AutoTokenizer.from_pretrained(ENGLISH_MODEL_PATH, local_files_only=True)
model_en = AutoModelForSequenceClassification.from_pretrained(ENGLISH_MODEL_PATH, local_files_only=True)
english_classifier = pipeline("text-classification", model=model_en, tokenizer=tokenizer_en, return_all_scores=True)

# 🔁 Load Hinglish model (local only)
print("🔁 Loading Hinglish model...")
tokenizer_hi = AutoTokenizer.from_pretrained(HINGLISH_MODEL_PATH, local_files_only=True)
model_hi = AutoModelForSequenceClassification.from_pretrained(HINGLISH_MODEL_PATH, local_files_only=True)
hinglish_classifier = pipeline("text-classification", model=model_hi, tokenizer=tokenizer_hi, return_all_scores=True)

# 📍 Detect language (simple logic)
def is_hinglish(text):
    hinglish_keywords = ["hai", "nahi", "kya", "bhaut", "acha", "mera", "apna"]
    return any(word in text.lower() for word in hinglish_keywords) or re.search(r"[\u0900-\u097F]", text)

# 🧠 Emotion detection interface
def detect_emotion(text, top_k=1):
    language = "hinglish" if is_hinglish(text) else "english"
    classifier = hinglish_classifier if language == "hinglish" else english_classifier
    result = classifier(text)
    top_emotions = sorted(result[0], key=lambda x: x['score'], reverse=True)[:top_k]
    return [e['label'].lower() for e in top_emotions], language
