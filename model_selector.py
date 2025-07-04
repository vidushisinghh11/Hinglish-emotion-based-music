from transformers import pipeline
import re

# Local model paths
ENGLISH_MODEL_PATH = "./english_model/emotion-music-model"
HINGLISH_MODEL_PATH = "./hinglish_model/hing-emotion-model"

# Load both models only once
print("🔁 Loading English model...")
english_classifier = pipeline(
    "text-classification",
    model=ENGLISH_MODEL_PATH,
    tokenizer=ENGLISH_MODEL_PATH,
    return_all_scores=True
)

print("🔁 Loading Hinglish model...")
hinglish_classifier = pipeline(
    "text-classification",
    model=HINGLISH_MODEL_PATH,
    tokenizer=HINGLISH_MODEL_PATH,
    return_all_scores=True
)

# Language detection (very basic)
def is_hinglish(text):
    # Heuristic: Hindi stopwords, Hinglish-style words, or Hindi/Devanagari letters
    hinglish_words = ["hai", "kya", "nahi", "bhaut", "acha", "sala", "mera", "apna"]
    if any(word in text.lower() for word in hinglish_words):
        return True

    # If Hindi script (Unicode range for Devanagari)
    if re.search("[\u0900-\u097F]", text):
        return True

    return False

# Emotion detection wrapper
def detect_emotion(text, top_k=1, return_list=False):
    language = "hinglish" if is_hinglish(text) else "english"
    classifier = hinglish_classifier if language == "hinglish" else english_classifier
    print(f"➡ Detected language: {language}")

    result = classifier(text)
    top_emotions = sorted(result[0], key=lambda x: x['score'], reverse=True)[:top_k]
    emotions = [e['label'].lower() for e in top_emotions]

    if return_list or top_k > 1:
        return emotions, language
    else:
        return emotions[0], language


#langdetect
