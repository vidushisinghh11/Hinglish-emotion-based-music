import random
from spotify_auth import get_spotify_client

# Emotion to genre mapping
emotion_to_genre = {
    "happy": "party",
    "sad": "sad",
    "angry": "rock",
    "surprise": "indie",
    "fear": "ambient",
    "neutral": "chill",
    "love": "romantic",
    "disgust": "dark"
}

def recommend_songs(emotion: str, language: str = "hindi"):
    """
    Recommend songs based on a detected emotion and preferred language.
    """
    sp = get_spotify_client()
    results = []

    modifiers = ["song", "track", "music", "feel", "vibes", "tune"]
    genre = emotion_to_genre.get(emotion.lower(), "pop")
    modifier = random.choice(modifiers)
    query = f"{genre} {language} {modifier}"

    offset = random.randint(0, 10)
    search = sp.search(q=query, type='track', limit=5, offset=offset)

    for item in search['tracks']['items']:
        name = item['name']
        artist = item['artists'][0]['name']
        url = item['external_urls']['spotify']
        results.append({
            "name": f"{name} by {artist}",
            "url": url
        })

    return results or [{"name": "No songs found", "url": "#"}]
