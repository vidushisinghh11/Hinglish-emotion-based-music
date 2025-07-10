from flask import Flask, render_template, request
from model_selector import detect_emotion
from recommender import recommend_songs

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    text = ""
    emotions = []
    songs = []
    model_choice = "hinglish"

    if request.method == 'POST':
        text = request.form['text']
        model_choice = request.form.get('model', 'hinglish')
        emotions = detect_emotion(text, model_choice)
        songs = recommend_songs(emotions[0][0], language="hindi" if model_choice == "hinglish" else "english")

    return render_template('index.html', text=text, emotions=emotions, songs=songs, model=model_choice)

if __name__ == '__main__':
    print("🎶 Flask app running...")
    app.run(debug=True)
