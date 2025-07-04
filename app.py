from flask import Flask, render_template, request, jsonify
from model_selector import detect_emotion
from recommender import recommend_songs

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        language = request.form.get('language')  # ✅ Get selected language

        # Detect emotion
        emotion, _ = detect_emotion(text)  # Or use just emotion = ...

        # Recommend songs based on emotion + language
        songs = recommend_songs(emotion, language)

        return render_template(
            'index.html',
            text=text,
            emotion=emotion,
            songs=songs,
            language=language   # ✅ Send language to HTML for the radio button
        )

    return render_template('index.html')




@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    text = data.get('text', '')
    language = data.get('language', 'english')

    if not text:
        return jsonify({"error": "Text input is required"}), 400

    emotion = detect_emotion(text)
    songs = recommend_songs(emotion, language)

    return jsonify({"emotion": emotion, "songs": songs, "language": language})

if __name__ == '__main__':
    print("🎶 Flask server running...")
    app.run(debug=True)
