from transformers import pipeline

# 🔍 Load the fine-tuned Hinglish emotion model
classifier = pipeline(
    "text-classification",
    model="./hing-emotion-model",         # ✅ Folder with your trained model
    tokenizer="./hing-emotion-model",
    return_all_scores=True
)

def detect_emotion(text):
    """
    Takes Hinglish input text and returns the top 2 predicted emotions.
    """
    result = classifier(text)
    print("Raw model output:", result)
    
    top_emotions = sorted(result[0], key=lambda x: x['score'], reverse=True)[:2]
    return [e['label'].lower() for e in top_emotions]
